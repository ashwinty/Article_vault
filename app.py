import streamlit as st
import main
import time

# Set page title and favicon
st.set_page_config(page_title="Article Vault", page_icon=":briefcase:")

# Initialize search history in session state
if 'search_history' not in st.session_state:
    st.session_state.search_history = []

# Main title
st.title("Article Vault")

# Input for query
query = st.text_input("Enter your query:")

# Search button
if st.button("Search"):
    # Display loading spinner while searching
    with st.spinner("Searching..."):
        # Pause for a short duration to simulate searching
        time.sleep(0)
        
        # Perform the search
        output = main.to_run(query)
        
        # Append query and response to search history
        st.session_state.search_history.append({'query': query, 'response': output})

        # Display search results
        for i, result in enumerate(output, 1):
            # Display case details
            st.markdown(f"### Result {i}")
            st.write("- **Title:**", result[0][0])
            st.write("- **Tag:**", result[0][1])
            st.write("- **Author:**", result[0][2])
            st.write("- **Date:**", result[0][3])
            st.write("- **Article URL:**", result[0][4])
            st.write("- **Description:**", result[0][5])
            st.write("- **Main Image URL:**", result[0][6])
            
            # Display highlighted text
            st.markdown(result[1], unsafe_allow_html=True)

# Display search history on the sidebar
st.sidebar.title("Search History")
if st.session_state.search_history:
    for i, item in enumerate(st.session_state.search_history):
        query = item['query']
        st.sidebar.markdown(f"**Query {i+1}:** {query}")
        if st.sidebar.button(f"View Results {i+1}"):
            response = item['response']
            st.sidebar.markdown("Response:")
            for j, result in enumerate(response, 1):
                st.sidebar.markdown(f"**Result {j}:**")
                st.sidebar.write("- **Title:**", result[0][0])
                st.sidebar.write("- **Tag:**", result[0][1])
                st.sidebar.write("- **Author:**", result[0][2])
                st.sidebar.write("- **Date:**", result[0][3])
                st.sidebar.write("- **Article URL:**", result[0][4])
                st.sidebar.write("- **Description:**", result[0][5])
                st.sidebar.write("- **Main Image URL:**", result[0][6])
                st.sidebar.markdown(result[1], unsafe_allow_html=True)
else:
    st.sidebar.markdown("No search history yet.")




# pip install llama-index-vector-stores-faiss  
# pip install langchain langchain-community langchain-core langchain-openai pandas transformers torch llama-index faiss-cpu   












# import streamlit as st
# import main
# import time

# # Set page title and favicon
# st.set_page_config(page_title="Article Vault", page_icon=":briefcase:")

# # Initialize list to store search history
# search_history = []

# # Main title
# st.title("Article Vault")

# # Input for query
# query = st.text_input("Enter your query:")

# # Search button
# if st.button("Search"):
#     # Display loading spinner while searching
#     with st.spinner("Searching..."):
#         # Pause for a short duration to simulate searching
#         time.sleep(1)
        
#         # Perform the semantic search
#         search_results = main.semantic_search(query)
        
#         # Perform the conversation with the agent
#         agent_response = main.handle_response(query, search_results, None)
        
#         # Append query to search history
#         search_history.append(query)

#         # Display search results
#         for i, result in enumerate(search_results, 1):
#             # Display case details
#             st.markdown(f"### Result {i}")
#             st.write("- **Title:**", result[0][0])
#             st.write("- **Tag:**", result[0][1])
#             st.write("- **Author:**", result[0][2])
#             st.write("- **Date:**", result[0][3])
#             st.write("- **Article URL:**", result[0][4])
#             st.write("- **Description:**", result[0][5])
#             st.write("- **Main Image URL:**", result[0][6])
            
#             # Display highlighted text
#             st.markdown(result[1], unsafe_allow_html=True)
            
#         # Display agent response
#         st.markdown("### Agent Response")
#         st.write(agent_response)

# # Display search history on the left of the screen
# st.sidebar.title("Search History")
# selected_query = st.sidebar.radio("Select a query:", search_history, index=len(search_history)-1 if search_history else None)











# import streamlit as st
# import main
# import time

# # Set page title and favicon
# st.set_page_config(page_title="Article Vault", page_icon=":briefcase:")

# # Initialize list to store search history
# search_history = []

# # Main title
# st.title("Article Vault")

# # Input for query
# query = st.text_input("Enter your query:")

# # Search button
# if st.button("Search"):
#     # Display loading spinner while searching
#     with st.spinner("Searching..."):
#         # Pause for a short duration to simulate searching
#         time.sleep(1)
        
#         # Perform the search
#         output = main.to_run(query)
        
#         # Append query to search history
#         search_history.append(query)

#         # Display search results
#         for i, result in enumerate(output, 1):
#             # Display case details
#             st.markdown(f"### Result {i}")
#             st.write("- **Title:**", result[0][0])
#             st.write("- **Tag:**", result[0][1])
#             st.write("- **Author:**", result[0][2])
#             st.write("- **Date:**", result[0][3])
#             st.write("- **Article URL:**", result[0][4])
#             st.write("- **Description:**", result[0][5])
#             st.write("- **Main Image URL:**", result[0][6])
            
#             # Display highlighted text
#             st.markdown(result[1], unsafe_allow_html=True)

# # Display search history on the left of the screen
# st.sidebar.title("Search History")
# selected_query = st.sidebar.radio("Select a query:", search_history, index=len(search_history)-1 if search_history else None)