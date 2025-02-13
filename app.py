import asyncio
import os
import json
import streamlit as st

from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion
from semantic_kernel.functions import KernelArguments
from semantic_kernel.connectors.ai.prompt_execution_settings import PromptExecutionSettings
from semantic_kernel.connectors.ai.function_choice_behavior import FunctionChoiceBehavior
from semantic_kernel.core_plugins import WebSearchEnginePlugin
from semantic_kernel.connectors.search_engine import BingConnector


# 🔹 Shopping Agent (Search Agent)
class ShoppingAgent:
    def __init__(self, kernel, bing_api_key):
        self.kernel = kernel
        self.bing_connector = BingConnector(bing_api_key)
        self.kernel.add_plugin(WebSearchEnginePlugin(self.bing_connector), "WebSearch")

    async def search_clothing(self, query):
        """
        Searches for clothing items using Bing API.

        Args:
            query (str): The search term (e.g., "men's waterproof winter jacket").

        Returns:
            list: A list of top search results with title, URL, and snippet.
        """
        try:
            # Enhanced query to focus on shopping results
            enhanced_query = f"buy {query} site:amazon.com OR site:ebay.com"
            search_results = await self.bing_connector.search(enhanced_query)

            # DEBUG: Print type and raw response
            print("Type of search_results:", type(search_results))
            print("Raw Bing API Response:", search_results)

            # If the response is a string, attempt to parse it as JSON
            if isinstance(search_results, str):
                try:
                    search_results = json.loads(search_results)
                except json.JSONDecodeError:
                    return [{"error": "Bing API returned unstructured data."}]

            # If search_results is a list, treat it as unstructured data
            if isinstance(search_results, list):
                return [{"error": f"Unexpected response format: {search_results}"}]

            # Ensure we have structured search results
            web_results = search_results.get("webPages", {}).get("value", [])

            formatted_results = [
                {
                    "title": item.get("name", "No Title"),
                    "url": item.get("url", "#"),
                    "snippet": item.get("snippet", "No description available."),
                }
                for item in web_results[:5]  # Limit results to top 5
            ]

            return formatted_results if formatted_results else [{"error": "No structured results found."}]

        except Exception as e:
            return [{"error": f"Failed to search: {str(e)}"}]




# 🔹 Main Streamlit App
async def main():
    # Initialize Streamlit UI
    st.set_page_config(page_title="👨‍🏫 Content Agents", layout="wide")
    st.title('👨‍🏫 Content Agents')

    # Sidebar for API Keys
    st.sidebar.title('🔑 Application Keys')
    azure_openai_key = st.sidebar.text_input('Azure OpenAI Key', 'edae3ed29311421a924559ebf25d2834')
    azure_openai_endpoint = st.sidebar.text_input('Azure OpenAI Endpoint', 'https://oregon-aoai.openai.azure.com')
    azure_openai_version = st.sidebar.text_input('Azure OpenAI Version', '2024-08-01-preview')
    azure_openai_deployment_name = st.sidebar.text_input('Azure OpenAI Deployment Name', 'gpt-4o-mini')
    bing_api_key = st.sidebar.text_input('Bing API Key', '69688f5fdb1449bcbdea6eae19edf31e')

    # Define the Kernel
    kernel = Kernel()

    # Add Azure OpenAI Connector
    chat_completion_service = AzureChatCompletion(
        deployment_name=azure_openai_deployment_name,
        api_key=azure_openai_key,
        endpoint=azure_openai_endpoint,
        api_version=azure_openai_version,
    )
    kernel.add_service(chat_completion_service)

    # Initialize Shopping Agent
    shopping_agent = ShoppingAgent(kernel, bing_api_key)

    # Execution settings for OpenAI model
    arguments = KernelArguments(
        settings=PromptExecutionSettings(
            function_choice_behavior=FunctionChoiceBehavior.Auto(),
        )
    )

    # UI Form for User Input
    with st.form('ai_form'):
        search_option = st.radio("Choose what to search for:", ["Clothing Recommendations", "Shop for Clothing Items"])
        text = st.text_area("Enter your query:", "What should I wear in Seattle in the spring?")

        submit = st.form_submit_button("Submit")
        if submit:
            if search_option == "Clothing Recommendations":
                # Use OpenAI to provide clothing recommendations
                response = await kernel.invoke_prompt(text, arguments=arguments)
                st.write(response.value)

            elif search_option == "Shop for Clothing Items":
                # Use Bing API to search for purchasable clothing items
                search_results = await shopping_agent.search_clothing(text)
                for result in search_results:
                    if "error" in result:
                        st.error(result["error"])
                    else:
                        st.markdown(f"**[{result['title']}]({result['url']})**")
                        st.write(result["snippet"])
                        st.write("---")


# Run the app
if __name__ == "__main__":
    asyncio.run(main())