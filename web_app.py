import os
from json import JSONEncoder
import time
import numpy as np
import datetime

import httpagentparser  # for getting the user agent as json
from flask import Flask, render_template, session, redirect, url_for
from flask import request

from myapp.analytics.analytics_data import AnalyticsData, ClickedDoc
from myapp.search.load_corpus import load_corpus
from myapp.search.objects import Document, StatsDocument
from myapp.search.search_engine import SearchEngine
from myapp.generation.rag import RAGGenerator, format_rag_response
from dotenv import load_dotenv
load_dotenv()  # take environment variables from .env


# *** for using method to_json in objects ***
def _default(self, obj):
    return getattr(obj.__class__, "to_json", _default.default)(obj)
_default.default = JSONEncoder().default
JSONEncoder.default = _default
# end lines ***for using method to_json in objects ***


# instantiate the Flask application
app = Flask(__name__)

# random 'secret_key' is used for persisting data in secure cookie
app.secret_key = os.getenv("SECRET_KEY")
# open browser dev tool to see the cookies
app.session_cookie_name = os.getenv("SESSION_COOKIE_NAME")
# instantiate our search engine
search_engine = SearchEngine()
# instantiate our in memory persistence
analytics_data = AnalyticsData()
# instantiate RAG generator
rag_generator = RAGGenerator()

# load documents corpus into memory.
full_path = os.path.realpath(__file__)
path, filename = os.path.split(full_path)
file_path = path + "/" + os.getenv("DATA_FILE_PATH")
corpus = load_corpus(file_path)
# Log first element of corpus to verify it loaded correctly:
print("\nCorpus is loaded... \n First element:\n", list(corpus.values())[0])


#Load of the inverted index and related data structures
start_time = time.time()
import gzip, pickle
from pathlib import Path
doc_lengths = len(corpus)
final_path = Path("data/index_snapshot.pkl.gz")
with gzip.open(final_path, "rb") as f:
    inv_index, tf, df, idf, title_index = pickle.load(f)
print("Total time to load the index: {} seconds" .format(np.round(time.time() - start_time, 2)))

# Home URL "/"
@app.route('/')
def index():
    print("starting home url /...")

    # flask server creates a session by persisting a cookie in the user's browser.
    # the 'session' object keeps data between multiple requests. Example:
    session['some_var'] = "Some value that is kept in session"
    session["queries"] = []  # initialize empty list of queries
    session["start_time"] = str(datetime.datetime.now())

    user_agent = request.headers.get('User-Agent')
    print("Raw user browser:", user_agent)

    user_ip = request.remote_addr
    agent = httpagentparser.detect(user_agent)
    session['user_agent'] = agent # This mistakes Windows 11 for Windows 10
    session['user_ip'] = user_ip

    print("Remote IP: {} - JSON user browser {}".format(user_ip, agent))
    print(session)
    return render_template('index.html', page_title="Welcome")


@app.route('/search', methods=['POST'])
def search_form_post():
    
    search_query = request.form['search-query']

    session['last_search_query'] = search_query
    session["queries"].append(search_query)

    search_id = analytics_data.save_query_terms(search_query)
    # Use Post-Redirect-Get pattern: store search info in session and redirect to GET results route
    session['last_search_id'] = search_id
    return redirect(url_for('results'))


@app.route('/results', methods=['GET'])
def results():
    # Render search results on GET to avoid browser POST resubmission when navigating back
    search_query = session.get('last_search_query')
    search_id = session.get('last_search_id')
    if not search_query:
        # nothing to show, redirect to home
        return redirect(url_for('index'))
    # compute full ranked results
    results = search_engine.search(search_query, search_id, corpus, inv_index, idf, tf, title_index, doc_lengths)

    # pagination parameters (page from query string)
    try:
        page = int(request.args.get('page', 1))
    except Exception:
        page = 1
    per_page = 20
    total_results = len(results)
    total_pages = max(1, int(np.ceil(total_results / per_page)))
    # clamp page
    if page < 1:
        page = 1
    if page > total_pages:
        page = total_pages

    start = (page - 1) * per_page
    end = start + per_page
    page_results = results[start:end]

    # generate RAG response based on user query and retrieved results
    rag_response = rag_generator.generate_response(search_query, results)
    print("RAG response:", rag_response)

    session['last_found_count'] = total_results

    # Record visit to results page for this query
    analytics_data.record_results_visit(search_query)
    
    # Record pagination behavior (track when users go beyond page 1)
    analytics_data.record_pagination(search_query, page)

    print(session)

    return render_template('results.html', results_list=page_results, page_title="Results", found_counter=total_results, rag_response=format_rag_response(rag_response), page=page, total_pages=total_pages, search_query=search_query)


@app.route('/doc_details', methods=['GET'])
def doc_details():
    """
    Show document details page
    ### Replace with your custom logic ###
    """

    # getting request parameters:
    # user = request.args.get('user')
    print("doc details session: ")
    print(session)

    res = session["some_var"]
    print("recovered var from session:", res)

    # get the query string parameters from request
    clicked_doc_id = request.args.get("pid")
    print("click in id={}".format(clicked_doc_id))

    # retrieve document from corpus if available
    doc = None
    if clicked_doc_id and clicked_doc_id in corpus:
        doc = corpus[clicked_doc_id]
    else:
        print(f"Document id {clicked_doc_id} not found in corpus")

    # store data in statistics table 1
    try:
        # Prevent double-counting when the browser issues duplicate quick requests
        last_pid = session.get('last_doc_viewed')
        last_time = session.get('last_doc_viewed_time', 0)
        now = time.time()
        search_query = session.get('last_search_query', 'direct')

        # Only count a new visit if it's a different pid or more than 5 seconds
        if clicked_doc_id and (last_pid != clicked_doc_id or (now - last_time) > 5):
            click_key = (clicked_doc_id, search_query)
            if click_key in analytics_data.fact_clicks:
                analytics_data.fact_clicks[click_key] += 1
            else:
                analytics_data.fact_clicks[click_key] = 1
            session['last_doc_viewed'] = clicked_doc_id
            session['last_doc_viewed_time'] = now
        else:
            print(f"Skipping duplicate increment for {clicked_doc_id}")
    except Exception as e:
        print('Error updating fact_clicks:', e)

    click_key = (clicked_doc_id, session.get('last_search_query', 'direct'))
    print(f"fact_clicks count for {click_key} is {analytics_data.fact_clicks.get(click_key, 0)}")
    print(analytics_data.fact_clicks)
    return render_template('doc_details.html', clicked_doc_id=clicked_doc_id, doc=doc, page_title="Document Details")


@app.route('/analytics/record_time', methods=['POST'])
def record_time():
    """
    Endpoint to receive time-spent analytics from the client.
    Expects JSON payload: { 'pid': '<doc_id>', 'seconds': <float> }
    """
    try:
        payload = request.get_json(force=True)
    except Exception as e:
        print('Invalid JSON payload for record_time:', e)
        return ("Bad Request", 400)

    if not payload:
        return ("Bad Request", 400)

    pid = payload.get('pid')
    seconds = payload.get('seconds')
    if not pid or seconds is None:
        return ("Missing pid or seconds", 400)

    search_query = session.get('last_search_query', 'direct')
    res = analytics_data.record_time_spent(pid, seconds, search_query)
    print(f"Recorded time for ({pid}, {search_query}): {seconds} seconds. Aggregated: {res}")
    return ({'status': 'ok', 'aggregated': res}, 200)


@app.route('/stats', methods=['GET'])
def stats():
    """
    Show simple statistics example. ### Replace with yourdashboard ###
    :return:
    """

    docs = []
    for (doc_id, query), count in analytics_data.fact_clicks.items():
        if doc_id in corpus:
            row: Document = corpus[doc_id]
            doc = StatsDocument(pid=row.pid, title=row.title, description=row.description, url=row.url, count=count)
            doc.query = query  # Store query info in doc
            docs.append(doc)
    
    # simulate sort by ranking
    docs.sort(key=lambda doc: doc.count, reverse=True)
    return render_template('stats.html', clicks_data=docs, agent=session.get('user_agent'), ip=session.get('user_ip'), date=session.get('start_time'))


@app.route('/dashboard', methods=['GET'])
def dashboard():
    visited_docs = []
    for (doc_id, query), count in analytics_data.fact_clicks.items():
        if doc_id in corpus:
            d: Document = corpus[doc_id]
            doc = ClickedDoc(doc_id, d.description, count, query)
            visited_docs.append(doc)

    # simulate sort by ranking
    visited_docs.sort(key=lambda doc: doc.counter, reverse=True)

    for doc in visited_docs: print(doc)

    # Build time spent statistics (if any)
    time_stats = []
    for (doc_id, query), t in analytics_data.fact_time_spent.items():
        if doc_id in corpus:
            d = corpus[doc_id]
            title = d.title if d else doc_id
            total_seconds = t.get('total_seconds', 0) if isinstance(t, dict) else float(t)
            visits = t.get('visits', 0) if isinstance(t, dict) else 1
            avg_seconds = (total_seconds / visits) if visits else 0
            time_stats.append({
                'pid': doc_id,
                'title': title,
                'query': query,
                'total_seconds': round(total_seconds, 2),
                'visits': visits,
                'avg_seconds': round(avg_seconds, 2),
                'url': d.url if d else '#'
            })

    # sort by total time spent desc
    time_stats.sort(key=lambda r: r['total_seconds'], reverse=True)

    # Get CTR statistics (product clicks vs doc views)
    ctr_stats = analytics_data.get_ctr_stats()
    # Enrich with document titles
    for stat in ctr_stats:
        doc_id = stat['doc_id']
        if doc_id in corpus:
            stat['title'] = corpus[doc_id].title
        else:
            stat['title'] = doc_id

    return render_template('dashboard.html', visited_docs=visited_docs, time_stats=time_stats, ctr_stats=ctr_stats, page_title="Dashboard")


# New route added for generating an examples of basic Altair plot (used for dashboard)
@app.route('/plot_number_of_views', methods=['GET'])
def plot_number_of_views():
    return analytics_data.plot_number_of_views()


@app.route('/analytics/record_click', methods=['POST'])
def record_click():
    """
    Endpoint to receive click coordinates from the client for heatmap visualization.
    Expects JSON payload: { 'x': <float>, 'y': <float>, 'page': '<page_id>' }
    """
    try:
        payload = request.get_json(force=True)
    except Exception as e:
        print('Invalid JSON payload for record_click:', e)
        return ("Bad Request", 400)

    if not payload:
        return ("Bad Request", 400)

    x = payload.get('x')
    y = payload.get('y')
    page = payload.get('page', 'unknown')
    
    if x is None or y is None:
        return ("Missing x or y", 400)

    res = analytics_data.record_click_heatmap(x, y, page)
    return ({'status': 'ok', 'click': res}, 200)


@app.route('/plot_click_heatmap', methods=['GET'])
def plot_click_heatmap():
    page = request.args.get('page')
    return analytics_data.plot_click_heatmap(page=page)


@app.route('/plot_results_visits', methods=['GET'])
def plot_results_visits():
    return analytics_data.plot_results_visits()


@app.route('/plot_pagination_queries', methods=['GET'])
def plot_pagination_queries():
    return analytics_data.plot_pagination_queries()


@app.route('/analytics/record_product_click', methods=['POST'])
def record_product_click():
    """
    Endpoint to receive product link clicks from doc_details page.
    Expects JSON payload: { 'pid': '<doc_id>' }
    """
    try:
        payload = request.get_json(force=True)
    except Exception as e:
        print('Invalid JSON payload for record_product_click:', e)
        return ("Bad Request", 400)

    if not payload:
        return ("Bad Request", 400)

    pid = payload.get('pid')
    if not pid:
        return ("Missing pid", 400)

    search_query = session.get('last_search_query', 'direct')
    res = analytics_data.record_product_click(pid, search_query)
    print(f"Recorded product click for ({pid}, {search_query}). Total: {res}")
    return ({'status': 'ok', 'count': res}, 200)


if __name__ == "__main__":
    app.run(port=8088, host="0.0.0.0", threaded=False, debug=os.getenv("DEBUG"))
