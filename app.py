import streamlit as st
import pandas as pd
pip install scipy
pip install sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import coo_matrix
import numpy as np
import re
import pickle

# Load titles DataFrame from a pkl file
titles = pd.read_pickle('titles.pkl')

# Initialize the vectorizer globally
vectorizer = TfidfVectorizer()
Tfidf = vectorizer.fit_transform(titles['mod_title'])

# Initialize Liked books DataFrame in session state
if 'liked_books' not in st.session_state:
    st.session_state.liked_books = pd.DataFrame(columns=titles.columns)
    
def show_image(image_url, title, avg_rating, goodreads_url):
    st.image(image_url, caption=f"{title}\nAverage Rating: {avg_rating}", use_column_width=True, width=100)
    st.markdown(f"<p style='text-align: center;'><a href='{goodreads_url}' target='_blank'>Read More...</a></p>", unsafe_allow_html=True)

# Function to perform book search
def search(query, titles):
    global vectorizer
    
    processed = re.sub("[^a-zA-Z0-9 ]", "", query.lower())
    query_vec = vectorizer.transform([processed])
    similarity = cosine_similarity(query_vec, vectorizer.transform(titles['mod_title'])).flatten()
    indices = np.argpartition(similarity, -10)[-10:]
    results = titles.iloc[indices]
    results = results.sort_values('ratings', ascending=False)

    # Display top search result and add to selected
    st.subheader("Top Result:")
    st.write(results.head(1))

    # Automatically select the first result
    selected_result = results.head(1)

    # Add selected result to Liked books in session state
    st.session_state.liked_books = st.session_state.liked_books._append(selected_result, ignore_index=True)

    # Display the selected result
    st.success("Treasure was successfully claimed!")

    # Return the results DataFrame
    return results.head(1)

# Function to view Liked books
def view_liked_books():
    if not st.session_state.liked_books.empty:
        st.subheader("Claimed Treasure! :gem:")
        st.balloons()
        st.write(st.session_state.liked_books)
    else:
        st.subheader("You have no treasure yet!")

# Function to save Liked books to a pickle file
def save_liked_books():
    st.session_state.liked_books.to_pickle('likedbooks.pkl')
    
# Function to display top N books from titles DataFrame, sorted by average rating
def view_top_books(n=100):
    top_books = titles.sort_values(by=['ratings', 'average_rating'], ascending=[False, False]).head(n)
    st.write(top_books)

def transform_liked_books(liked_books):
    """
    Transform the liked books DataFrame into a new DataFrame with columns: user_id, book_id, rating, title.

    Parameters:
    - liked_books (pd.DataFrame): DataFrame containing liked books information.

    Returns:
    - pd.DataFrame: Transformed DataFrame with user_id=-1, book_id, rating (average_rating), and title.
    """
    transformed_data = {
        'user_id': [-1] * len(liked_books),
        'book_id': liked_books['book_id'].tolist(),
        'rating': liked_books['average_rating'].tolist(),
        'title': liked_books['title'].tolist()
    }

    my_books = pd.DataFrame(transformed_data)
    return my_books

def generate_recommendations(my_books, titles):
    #rec = pd.read_csv('recommendation.csv')
    #return rec
  
   # import the liked list 
    my_books["book_id"] = my_books["book_id"].astype(str)
    #stream through the user book mapping 
    csv_book_mapping = {}

    with open("book_id_map.csv", "r") as f:
        while True:
            line = f.readline()
            if not line:
                break
            csv_id, book_id = line.strip().split(",")
            csv_book_mapping[csv_id] = book_id
    #create my book set 
    book_set = set(my_books["book_id"])
    # create ovelapping user set 

    overlap_users = {}

    with open("goodreads_interactions.csv", 'r') as f:
        while True:
            line = f.readline()
            if not line:
                break
            user_id, csv_id, _, rating, _ = line.split(",")

            book_id = csv_book_mapping.get(csv_id)

            if book_id in book_set:
                if user_id not in overlap_users:
                    overlap_users[user_id] = 1
                else:
                    overlap_users[user_id] += 1
    # make a filtered list to match my liked books and ratings 
    filtered_overlap_users = set([k for k in overlap_users if overlap_users[k] > my_books.shape[0]/5])
    len(filtered_overlap_users)
    # create a list for the interactions 
    interactions_list = []

    with open("goodreads_interactions.csv", 'r') as f:
        while True:
            line = f.readline()
            if not line:
                break
            user_id, csv_id, _, rating, _ = line.split(",")

            if user_id in filtered_overlap_users:
                book_id = csv_book_mapping[csv_id]
                interactions_list.append([user_id, book_id, rating])
    # append our liked list to the interractions list 
    interactions = pd.DataFrame(interactions_list, columns=["user_id", "book_id", "rating"])
    interactions = pd.concat([my_books[["user_id", "book_id", "rating"]], interactions])
    # checking the datatypes are aligned 
    interactions["book_id"] = interactions["book_id"].astype(str)
    interactions["user_id"] = interactions["user_id"].astype(str)
    interactions["rating"] = pd.to_numeric(interactions["rating"])
    # getting index to build matrix 
    interactions["user_index"] = interactions["user_id"].astype("category").cat.codes
    interactions["book_index"] = interactions["book_id"].astype("category").cat.codes
    # creating the matrix and viewing our user information
    ratings_mat_coo = coo_matrix((interactions["rating"], (interactions["user_index"], interactions["book_index"])))
    ratings_mat = ratings_mat_coo.tocsr()
    interactions[interactions["user_id"] == "-1"]
    my_index = 0 
    similarity = cosine_similarity(ratings_mat[my_index,:], ratings_mat).flatten()
    indices = np.argpartition(similarity, -15)[-15:]
    similar_users = interactions[interactions["user_index"].isin(indices)].copy()
    similar_users = similar_users[similar_users["user_id"]!="-1"]
    book_recs = similar_users.groupby("book_id").rating.agg(['count', 'mean'])
    book_recs = book_recs.merge(titles, how="inner", on="book_id")
    book_recs["adjusted_count"] = book_recs["count"] * (book_recs["count"] / book_recs["ratings"])
    book_recs["score"] = book_recs["mean"] * book_recs["adjusted_count"]
    book_recs = book_recs[~book_recs["book_id"].isin(my_books["book_id"])]
    my_books["mod_title"] = my_books["title"].str.replace("[^a-zA-Z0-9 ]", "", regex=True).str.lower()
    my_books["mod_title"] = my_books["mod_title"].str.replace("\s+", " ", regex=True)
    book_recs = book_recs[~book_recs["mod_title"].isin(my_books["mod_title"])]
    book_recs = book_recs[book_recs["mean"] >=4]
    book_recs = book_recs[book_recs["count"]>2]
    top_recs = book_recs.sort_values("mean", ascending=False)
    return top_recs
    

# Streamlit app
st.title("Welcome to Read Radar :book:")

# Sidebar navigation
page = st.sidebar.selectbox("Navigation:",["Radar Treasure","Explore", "My Chest", "Scan Radar"])

# Display content based on the selected page
if page == "Radar Treasure":
    st.title("The Top 100 Books :books:")
    # Display top books with cover images
    cols = st.columns(5)
    for _, row in titles.sort_values(by=['ratings', 'average_rating'], ascending=[False, False]).head(100).iterrows():
        col = cols[int(row.name % 5)]
        with col:
            show_image(row['cover_image'], row['title'], row['average_rating'], row['url'])


elif page == "Explore":
    st.title("Explore Books :mag:")
    # User input for book search
    search_query = st.text_input("Enter your search query:")
    # Search button
    if st.button("Claim!"):
        st.subheader("Top Search Results:")
         # Display search results with cover images
        cols = st.columns(5)
        for _, row in search(search_query, titles).iterrows():
            col = cols[int(row.name % 5)]
            with col:
                show_image(row['cover_image'], row['title'], row['average_rating'], row['url'])

elif page == "My Chest":
    st.title("Your Liked Books :gem:")
    # Display liked books with cover images
    cols = st.columns(5)
    for _, row in st.session_state.liked_books.iterrows():
        col = cols[int(row.name % 5)]
        with col:
            show_image(row['cover_image'], row['title'], row['average_rating'], "")
            
            
elif page == "Scan Radar":
    st.title("Your Radar Picks :satellite:")
    st.subheader("Books Recommended Just For You!")
    # For demonstration purposes, let's assume 'rec' is the DataFrame with recommended books
    rec = generate_recommendations(transform_liked_books(st.session_state.liked_books), titles)
    st.write(rec)
    #Display recommended books with cover images
    cols = st.columns(5)
    for _, row in rec.iterrows():
        col = cols[int(row.name % 5)]
        with col:
            show_image(row['cover_image'], row['title'], row['average_rating'], row['url'])
