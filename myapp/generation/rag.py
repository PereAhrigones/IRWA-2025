import os
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
        - AVAILABILITY STATUS
        Explain: Number of in-stock vs out-of-stock items

        3. 
        - TOP 3 RECOMMENDATIONS
        Rank products considering: query relevance, price, rating, and availability:
        1. **Best Overall**: [PID] - [Title] - [Key reasons: price, rating, relevance]
        2. **Best Value**: [PID] - [Title] - [Best discount-to-quality ratio]
        3. **Premium Choice**: [PID] - [Title] - [Highest rated if budget allows]

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
                 Average Rating: {res.avg_rating}, \
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
    

def format_rag_response(generation: str) -> str:
    """
    Format RAG response for HTML display with proper formatting.
    """
    if not generation:
        return ""
    
    # Add line breaks before each bullet point
    lines = generation.split('\n')
    formatted_lines = []
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # Format bullet points with bold labels
        if line.startswith('-'):
            # Find the colon position
            colon_pos = line.find(':')
            if colon_pos > 0:
                # Split into label and content
                label = line[:colon_pos]
                content = line[colon_pos + 1:].strip()
                
                # Format as: <strong>label</strong>: content
                formatted = f"<strong>{label}</strong>: {content}"
                formatted_lines.append(formatted)
            else:
                formatted_lines.append(f"<strong>{line}</strong>")
        else:
            # For non-bullet text, just add as paragraph
            formatted_lines.append(line)
    
    # Join with line breaks
    return '<br>'.join(formatted_lines)