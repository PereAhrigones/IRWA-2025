import os
import re
from groq import Groq
from dotenv import load_dotenv
load_dotenv()  # take environment variables from .env


class RAGGenerator:

    PROMPT_TEMPLATE = """
        You are an expert e-commerce analyst providing comprehensive search result summaries. 
        Analyze the following products and create a useful summary for the shopper.

        ## ANALYSIS REQUIREMENTS:
        Provide a structured summary with these sections:

        1. 
        - QUERY MATCH ASSESSMENT
        Explain: How well do results match the query "{user_query}"?, Grade: Excellent/Good/Fair/Poor with brief explanation

        2. 
        - PRICING LANDSCAPE
        Explain: Price range found: [Lowest] to [Highest], Average discount percentage across products, and nothing more, do not give me any product/document

        3. 
        - AVAILABILITY STATUS
        Explain: Number of in-stock vs out-of-stock items, if it is false that X products are out of stock Say "X products are available", else Say "X products are out of Stock", and nothing more

        4. 
        - QUALITY INDICATORS
        Explain: Average rating across all products, Number of good rated products (rating ≥ 4.0), Number of products with concerning ratings (rating < 3.0), just the numbers and nothing more, do not give me any product/document

        5. 
        - TOP 3 RECOMMENDATIONS
        Here it is the time to show products/documents to the user.
        Rank products considering: query relevance, price, rating, and availability:
        1. **Best Overall**: [PID] - [Title] - [Using all the info of the products/documents, short explanation of why this is the best document]
        2. **Best Value**: [PID] - [Title] - [Best discount-to-quality ratio]
        3. **Premium Choice**: [PID] - [Title] - [Highest rated if budget allows]

        6. 
        - SEARCH IMPROVEMENTS
        Explain: (Create this section if needed, else do not show it) Suggested refined queries for better results, What might be missing from current results. Do not textually use product/document names as new queries.

        ## IMPORTANT FORMATTING RULES:
        - Use bullet points for lists
        - Include specific numbers (prices, ratings, counts)
        - Reference products by [PID] [Name/Title] for clarity
        - Keep advice practical and actionable

        ## PRODUCT DATA:
        {retrieved_results}

        ## YOUR ANALYSIS SUMMARY:
    """

    def generate_response(self, user_query: str, retrieved_results: list, top_N: int = 20) -> dict:
        """
        Generate a response using the retrieved search results. 
        Returns:
            dict: Contains the generated suggestion and the quality evaluation.
        """
        DEFAULT_ANSWER = "RAG is not available. Check your credentials (.env file) or account limits."
        try:
            client = Groq(
                api_key=os.environ.get("GROQ_API_KEY"),
            )
            model_name = os.environ.get("GROQ_MODEL", "llama-3.1-8b-instant")

            # Format the retrieved results for the prompt
            formatted_results = "\n\n".join(
                [f"- PID: {res.pid}, Title: {res.title}, Original Price: {res.actual_price}, Price with applied discount: {res.selling_price}, \
                 Average Rating: {res.average_rating}, \
                 It is {res.out_of_stock} that the product is out of stock. \
                 Description of the product: {res.description}" for res in retrieved_results[:top_N]]
            )

            prompt = self.PROMPT_TEMPLATE.format(
                retrieved_results=formatted_results,
                user_query=user_query
            )

            chat_completion = client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                model=model_name,
            )

            generation = chat_completion.choices[0].message.content


            return generation
        except Exception as e:
            print(f"Error during RAG generation: {e}")
            return DEFAULT_ANSWER


    def visually_appealing_text(generation):
        if not generation:
            return ""
        
        # Add line break before each "-" that starts a line or follows a line break
        formatted = generation.replace('\n-', '\n<br>-')
        
        # Ensure the first "-" also gets a line break if at the beginning
        if formatted.startswith('-'):
            formatted = '<br>' + formatted
        
        # Make text between "-" and ":" bold
        lines = formatted.split('\n')
        result_lines = []
        
        for line in lines:
            # Check if line contains "-" followed by ":" somewhere in the line
            if '-' in line and ':' in line:
                # Find the position of "-" and the next ":"
                dash_pos = line.find('-')
                colon_pos = line.find(':', dash_pos)
                
                if dash_pos < colon_pos:
                    # Extract parts
                    before_dash = line[:dash_pos]
                    between = line[dash_pos:colon_pos]
                    after_colon = line[colon_pos:]
                    
                    # Format with HTML
                    formatted_line = f"{before_dash}<strong>{between}</strong>{after_colon}"
                    result_lines.append(formatted_line)
                else:
                    result_lines.append(line)
            else:
                result_lines.append(line)
        
        return '<br>'.join(result_lines)
    

import re  # Add this import

def format_rag_response(generation: str) -> str:
    """
    Format RAG response for HTML display with proper formatting.
    Handles: ## Sections, ### Subsections, * Bullets, - Bullets, **Bold**, PID references
    """
    if not generation:
        return ""
    
    lines = generation.split('\n')
    formatted_lines = []
    in_section = False
    
    for line in lines:
        line = line.strip()
        if not line:
            if in_section:  # Add spacing between sections
                formatted_lines.append('<div class="section-spacer"></div>')
                in_section = False
            continue
        
        # Handle section headers (##)
        if line.startswith('## '):
            title = line[3:].strip()
            formatted_lines.append(f'<h4 class="rag-section">{title}</h4>')
            in_section = True
        
        # Handle subsection headers (###)
        elif line.startswith('### '):
            title = line[4:].strip()
            formatted_lines.append(f'<h5 class="rag-subsection">{title}</h5>')
            in_section = True
        
        # Handle bullet points (* or -)
        elif line.startswith('* ') or line.startswith('- '):
            content = line[2:].strip()
            
            # Check if it's a PID reference
            if 'PID:' in content or 'PID: ' in content:
                # Highlight PID references
                content = content.replace('PID:', '<span class="pid-highlight">PID:</span>')
                formatted_lines.append(f'<div class="rag-bullet pid-bullet">• {content}</div>')
            else:
                formatted_lines.append(f'<div class="rag-bullet">• {content}</div>')
        
        # Handle numbered lists (1., 2., etc.)
        elif re.match(r'^\d+\.\s', line):
            formatted_lines.append(f'<div class="rag-numbered">{line}</div>')
        
        # Handle bold text (**text**)
        elif '**' in line:
            # Replace all **text** with <strong>text</strong>
            bold_line = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', line)
            formatted_lines.append(f'<div class="rag-text">{bold_line}</div>')
        
        # Regular text
        else:
            # Check if line contains a colon (key: value pattern)
            if ': ' in line:
                parts = line.split(': ', 1)
                if len(parts) == 2:
                    key, value = parts
                    formatted_lines.append(f'<div class="rag-key-value"><span class="rag-key">{key}:</span> <span class="rag-value">{value}</span></div>')
                    continue
            
            formatted_lines.append(f'<div class="rag-text">{line}</div>')
    
    # Join all formatted lines
    return '\n'.join(formatted_lines)