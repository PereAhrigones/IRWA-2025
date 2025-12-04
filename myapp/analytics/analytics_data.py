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

    # Track click heatmap coordinates
    # Structure: list of {'x': float, 'y': float, 'page': str, 'timestamp': datetime}
    fact_clicks_heatmap = []

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

    def record_click_heatmap(self, x: float, y: float, page: str):
        """
        Record click coordinates for heatmap visualization.
        x, y: click coordinates (0-1 normalized or pixel values)
        page: page identifier (e.g., 'results', 'doc_details', 'search')
        """
        try:
            x = float(x)
            y = float(y)
        except (ValueError, TypeError):
            return None

        import datetime
        click_data = {
            'x': x,
            'y': y,
            'page': str(page),
            'timestamp': datetime.datetime.now().isoformat()
        }
        self.fact_clicks_heatmap.append(click_data)
        return click_data

    def plot_click_heatmap(self, page: str = None, width: int = 800, height: int = 600):
        """
        Generate a heatmap visualization of user clicks.
        Returns HTML with SVG heatmap or message if no data.
        """
        if not self.fact_clicks_heatmap:
            return """
            <html>
                <head><meta charset="utf-8"><title>Heatmap</title></head>
                <body style="font-family: Arial, sans-serif; color: #333;">
                    <div style="padding:20px; text-align:center;">
                        <h3>No click data available yet</h3>
                        <p>Click heatmap will appear here once users interact with the page.</p>
                    </div>
                </body>
            </html>
            """

        # Filter by page if specified
        clicks = [c for c in self.fact_clicks_heatmap if page is None or c.get('page') == page]
        
        if not clicks:
            return f"""
            <html>
                <head><meta charset="utf-8"><title>Heatmap</title></head>
                <body style="font-family: Arial, sans-serif; color: #333;">
                    <div style="padding:20px; text-align:center;">
                        <h3>No clicks recorded on page: {page}</h3>
                    </div>
                </body>
            </html>
            """

        # Create a simple heatmap using a grid and density
        # Bin clicks into a grid and color based on density
        grid_cols, grid_rows = 10, 8
        grid = [[0 for _ in range(grid_cols)] for _ in range(grid_rows)]

        for click in clicks:
            x = click.get('x', 0)
            y = click.get('y', 0)
            
            # Normalize if values are > 1 (pixel coordinates)
            if x > 1:
                x = x / width
            if y > 1:
                y = y / height
            
            # Clamp to [0, 1]
            x = max(0, min(1, x))
            y = max(0, min(1, y))
            
            # Map to grid
            col = int(x * (grid_cols - 1))
            row = int(y * (grid_rows - 1))
            grid[row][col] += 1

        # Find max for color scaling
        max_clicks = max(max(row) for row in grid) if any(any(row) for row in grid) else 1

        # Generate SVG
        cell_width = width / grid_cols
        cell_height = height / grid_rows
        
        svg_cells = []
        for row_idx, row in enumerate(grid):
            for col_idx, count in enumerate(row):
                x = col_idx * cell_width
                y = row_idx * cell_height
                
                # Color intensity based on clicks (red gradient)
                intensity = count / max_clicks if max_clicks > 0 else 0
                # RGB: from light red to dark red
                r = int(255)
                g = int(255 * (1 - intensity * 0.7))
                b = int(255 * (1 - intensity * 0.7))
                color = f'rgb({r},{g},{b})'
                opacity = 0.3 + (intensity * 0.7)
                
                svg_cells.append(f'''
                    <rect x="{x}" y="{y}" width="{cell_width}" height="{cell_height}" 
                          fill="{color}" opacity="{opacity}" stroke="#ccc" stroke-width="1"/>
                    <text x="{x + cell_width/2}" y="{y + cell_height/2}" 
                          text-anchor="middle" dominant-baseline="middle" 
                          font-size="10" fill="#333" opacity="0.5">{count if count > 0 else ''}</text>
                ''')

        svg_content = f'''
        <svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg" style="border: 1px solid #ccc;">
            {''.join(svg_cells)}
        </svg>
        '''

        return f"""
        <html>
            <head><meta charset="utf-8"><title>Click Heatmap</title></head>
            <body style="font-family: Arial, sans-serif; color: #333;">
                <div style="padding:20px;">
                    <h3>Click Heatmap {f'({page})' if page else ''}</h3>
                    <p>Total clicks: {len(clicks)}</p>
                    {svg_content}
                    <p style="margin-top:10px; font-size:0.9em; color:#666;">
                        Red areas indicate higher click density.
                    </p>
                </div>
            </body>
        </html>
        """


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
