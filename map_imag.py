import streamlit as st
import pandas as pd
import numpy as np
import leafmap.foliumap as leafmap
from folium.plugins import MarkerCluster
import folium
import tools_imag as ti
import dhlab.api.dhlab_api as api
import requests
from io import StringIO
from urllib.parse import quote
            
def dataframe_with_selections(df, key_prefix):
    df_with_selections = df.copy()
    df_with_selections.insert(0, "Select", False)
    edited_df = st.data_editor(
        df_with_selections,
        hide_index=True,
        column_config={"Select": st.column_config.CheckboxColumn(required=True)},
        disabled=df.columns,
        key=f"editor_{key_prefix}"
    )
    selected_indices = list(np.where(edited_df['Select'])[0])
    return selected_indices, edited_df

def hex_to_folium_color(hex_color):
    # Convert hex to RGB
    hex_color = hex_color.lstrip('#')
    r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    
    # Define basic Folium colors and their RGB values
    folium_colors = {
        'red': (255, 0, 0),
        'blue': (0, 0, 255),
        'green': (0, 128, 0),
        'purple': (128, 0, 128),
        'orange': (255, 165, 0),
        'darkred': (139, 0, 0),
        'lightred': (255, 128, 128),
        'darkblue': (0, 0, 139),
        'darkgreen': (0, 100, 0),
        'cadetblue': (95, 158, 160),
        'darkpurple': (148, 0, 211),
        'pink': (255, 192, 203),
        'lightblue': (173, 216, 230),
        'lightgreen': (144, 238, 144)
    }
    
    # Find closest color using Euclidean distance
    min_distance = float('inf')
    closest_color = 'blue'  # default
    
    for name, rgb in folium_colors.items():
        distance = sum((c1 - c2) ** 2 for c1, c2 in zip((r, g, b), rgb))
        if distance < min_distance:
            min_distance = distance
            closest_color = name
    
    return closest_color
@st.cache_data()
def sheet_display(df):
    term_counts = df[['Term']].groupby('Term').size().reset_index(name='Count')
    return term_counts

@st.cache_data()
def corpus():
    korpus_file = "imag_korpus.xlsx"
    corpus = pd.read_excel(korpus_file, index_col=0)
    authors = list(set(corpus.author))
    titles = list(set(corpus.title))
    categories = list(set(corpus.category))
    return corpus, authors, titles, categories

@st.cache_data
def load_exploded_places():
    df = pd.read_pickle('exploded_places.pkl')
    #st.write("Columns in df:", df.columns)
    return df


@st.cache_data()
def calculate_place_stats(places_data):
    """Calculate frequency and dispersion metrics for places"""
    # Group by place name to combine statistics
    place_stats = {}
    
    for name, group in places_data.groupby('name'):
        total_freq = group['frekv'].sum()
        docs = set(group['dhlabid'])
        dispersion = len(docs) / len(places_data['dhlabid'].unique())
        score = total_freq * dispersion
        
        place_stats[name] = {
            'freq': total_freq,
            'docs': docs,
            'dispersion': dispersion,
            'score': score,
            'lat': group['latitude'].iloc[0],
            'lon': group['longitude'].iloc[0],
            'token': group['token'].iloc[0]
        }
    
    return place_stats

def select_places(place_stats, min_score=0.01, max_places=200):
    """Select places based on combined frequency/dispersion score"""
    filtered = {p: s for p, s in place_stats.items() if s['score'] >= min_score}
    return dict(sorted(filtered.items(), 
                      key=lambda x: x[1]['score'], 
                      reverse=True)[:max_places])

# Define available basemaps
BASEMAP_OPTIONS = [
    "OpenStreetMap.Mapnik",
    "CartoDB.Positron",
    "CartoDB.DarkMatter",
   
]

st.set_page_config(layout="wide")
st.title("ImagiNation - kart")


preprocessed_places = load_exploded_places()


corpus_df, authorlist, titlelist, categorylist = corpus()
col1, col_map = st.columns([1, 3])

with col1:
    st.header("Bygg et korpus")
    subkorpus = corpus_df.copy()
    subkorpus['author'] = subkorpus['author'].fillna('').apply(lambda x: x.replace('/', ' '))

    categories = st.multiselect("Kategori", 
                              options=categorylist,
                              key="category_select")
    if categories != []:
        subkorpus = subkorpus[subkorpus.category.isin(categories)]

    
    authors = st.multiselect('Forfatter', 
                           options=list(set(subkorpus.author)),
                           key="author_select")
    if authors != []:
        subkorpus = subkorpus[subkorpus.author.isin(authors)]
    
    titles = st.multiselect('Tittel', 
                          options=list(set(subkorpus.title)),
                          key="title_select")
    if titles != []:
        subkorpus = subkorpus[subkorpus.title.isin(titles)]
        
    all_place_names = sorted(preprocessed_places['name'].unique())
    selected_places = st.multiselect(
        'Select places to include', 
        options=all_place_names,
        help="Show books that mention these places"
    )
    
    # If places selected, filter subkorpus
    if selected_places:
        # Get all dhlabids for selected places
        place_books = preprocessed_places[
            preprocessed_places['name'].isin(selected_places)
        ]['docs'].unique()
        
        subkorpus = subkorpus[subkorpus.dhlabid.isin(place_books)]
        st.write(f"Found {len(subkorpus)} books mentioning selected places")
    st.write(f"Antall bøker i korpus: {len(subkorpus)}")
    st.write(f"Antall forfattere: {len(list(set(subkorpus.author)))}")
    #st.dataframe(subkorpus)
    basemap = st.selectbox("Choose map style", BASEMAP_OPTIONS, index=BASEMAP_OPTIONS.index("OpenStreetMap.Mapnik"))
    max_places = st.slider(
        "Maximum number of places to show", 
        min_value=1, 
        max_value=500,
        value=200,
        step=10,
        help="Higher numbers may affect performance"
    )

# Add in the control section with other sliders
    max_books = st.slider(
        "Maximum number of books to process", 
        min_value=100, 
        max_value=5000,
        value=400,
        step=100,
        help="Lower numbers give faster response but less complete picture"
    )

# Then use this instead of hardcoded 2000
if len(subkorpus) > max_books:
    selected_dhlabids = subkorpus.sample(max_books).dhlabid
    st.write(f"Sampling {max_books} books from selection")
else:
    selected_dhlabids = subkorpus.dhlabid
    st.write(f"Using all {len(subkorpus)} books")

# Get places directly
places = ti.geo_locations_corpus(selected_dhlabids)
st.write(f"Place mentions found: {len(places)}")
places = places[places['rank']==1]

# Group places to get statistics
significant_places = (places.groupby('name')
    .agg({
        'token': 'first',
        'frekv': 'sum',
        'latitude': 'first',
        'longitude': 'first',
        'dhlabid': list
    })
    .reset_index()
)

# Calculate dispersion
total_books = len(selected_dhlabids)
significant_places['dispersion'] = significant_places['dhlabid'].apply(lambda x: len(x) / total_books)
significant_places['score'] = significant_places['frekv'] * significant_places['dispersion']

# Get top places
significant_places = significant_places.nlargest(max_places, 'score')


with col_map:
    # Use DataFrame columns directly
    center_lat = significant_places['latitude'].mean()
    center_lon = significant_places['longitude'].mean()
    
    m = leafmap.Map(
       center=(center_lat, center_lon),
       zoom=5,
       basemap=basemap
    )
    
    cluster = MarkerCluster().add_to(m)

    # Create markers using DataFrame rows
    for _, place in significant_places.iterrows():
        # Get books for this place
        place_books = corpus_df[corpus_df.dhlabid.isin(place['dhlabid'])]
        book_count = len(place_books)
        
        html = f"""
        <div style='width:500px'>
            <h4>{place['name']}</h4>
            <p><strong>Historical name:</strong> {place['token']}</p>
            <p><strong>{place['frekv']} mentions in {book_count} books</strong></p>
            <p>Dispersion score: {place['dispersion']:.3f}</p>
            <div style='max-height: 400px; overflow-y: auto;'>
                <table style='width: 100%; border-collapse: collapse;'>
                    <thead style='position: sticky; top: 0; background: white;'>
                        <tr>
                            <th style='border: 1px solid #ddd; padding: 8px; text-align: left;'>Title</th>
                            <th style='border: 1px solid #ddd; padding: 8px; text-align: left;'>Author</th>
                            <th style='border: 1px solid #ddd; padding: 8px; text-align: left;'>Year</th>
                        </tr>
                    </thead>
                    <tbody>
        """
        
        # Add book rows
        for _, book in place_books.iterrows():
            book_url = f"https://nb.no/items/{book.urn}?searchText=\"{quote(place['token'])}\""
            html += f"""
                <tr>
                    <td style='border: 1px solid #ddd; padding: 8px;'>
                        <a href='{book_url}' target='_blank'>{book.title}</a>
                    </td>
                    <td style='border: 1px solid #ddd; padding: 8px;'>{book.author}</td>
                    <td style='border: 1px solid #ddd; padding: 8px;'>{book.year}</td>
                </tr>
            """
        
        html += """
                    </tbody>
                </table>
            </div>
        </div>
        """
        
        folium.Marker(
            location=[place['latitude'], place['longitude']],
            popup=folium.Popup(html, max_width=500),
            icon=folium.Icon(color="red"),
            tooltip=f"{place['name']}: {place['frekv']} mentions in {book_count} books"
        ).add_to(cluster)
    
    folium.LayerControl().add_to(m)
    m.to_streamlit(height=600)