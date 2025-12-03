import json
import random
import altair as alt
import pandas as pd


class AnalyticsData:
    """
    An in memory persistence object.
    Declare more variables to hold analytics tables.
    """
    # Example of statistics table
    # fact_clicks is a dictionary with the click counters: key = doc id | value = click counter
    fact_clicks = dict([])

    # Track time spent per document (aggregated)
    # Structure: { doc_id: { 'total_seconds': float, 'visits': int } }
    fact_time_spent = dict([])

    ### Please add your custom tables here:

    def save_query_terms(self, terms: str) -> int:
        print(self)
        return random.randint(0, 100000)
    
    def plot_number_of_views(self):
        # Prepare data
        # If there are no clicks yet, return a simple HTML message instead of building a chart
        if not self.fact_clicks:
                return """
                <html>
                    <head><meta charset="utf-8"><title>No data</title></head>
                    <body style="font-family: Arial, sans-serif; color: #333;">
                        <div style="padding:20px; text-align:center;">
                            <h3>No analytics data available yet</h3>
                            <p>There are no document clicks recorded so far. The dashboard will show charts here once there is data.</p>
                        </div>
                    </body>
                </html>
                """

        data = [{'Document ID': doc_id, 'Number of Views': count} for doc_id, count in self.fact_clicks.items()]
        df = pd.DataFrame(data)
        # Create Altair chart
        chart = alt.Chart(df).mark_bar().encode(
                x='Document ID',
                y='Number of Views'
        ).properties(
                title='Number of Views per Document'
        )
        # Render the chart to HTML
        return chart.to_html()

    def record_time_spent(self, doc_id: str, seconds: float):
        """
        Record time spent by users on a document detail page.
        Aggregates total seconds and visit counts per document.
        """
        try:
            seconds = float(seconds)
        except Exception:
            return None

        if doc_id in self.fact_time_spent:
            entry = self.fact_time_spent[doc_id]
            entry['total_seconds'] += seconds
            entry['visits'] += 1
        else:
            self.fact_time_spent[doc_id] = {'total_seconds': seconds, 'visits': 1}

        return self.fact_time_spent[doc_id]


class ClickedDoc:
    def __init__(self, doc_id, description, counter):
        self.doc_id = doc_id
        self.description = description
        self.counter = counter

    def to_json(self):
        return self.__dict__

    def __str__(self):
        """
        Print the object content as a JSON string
        """
        return json.dumps(self)
