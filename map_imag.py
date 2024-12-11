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


# Define available basemaps
BASEMAP_OPTIONS = [
    "OpenStreetMap.Mapnik",
    "CartoDB.Positron",
    "CartoDB.DarkMatter",
   
]
basemap = "CartoDB.Positron"

st.set_page_config(layout="wide")
st.title("ImagiNation - kart")



corpus_df, authorlist, titlelist, categorylist = corpus()
col1, col_map = st.columns([1, 3])

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
    

    st.write(f"Antall bøker i korpus: {len(subkorpus)}")
    st.write(f"Antall forfattere: {len(list(set(subkorpus.author)))}")
    #st.dataframe(subkorpus)
    basemap = st.selectbox("Choose map style", BASEMAP_OPTIONS, index=BASEMAP_OPTIONS.index("OpenStreetMap.Mapnik"))

places_corpus = subkorpus.sample(min(10, len(subkorpus)))

places = ti.geo_locations_corpus(places_corpus.dhlabid)
places = places[places['rank']==1]
#st.write(len(places))
#st.dataframe(places)
st.dataframe(places[['token','name','frekv']].sort_values(by='frekv', ascending=False), hide_index=True)


with col_map:
    center_lat = places.latitude.mean()
    center_lon = places.longitude.mean()
    
    m = leafmap.Map(
        center=(center_lat, center_lon),
        zoom=5,
        max_zoom=18,
        min_zoom=2,
        control_scale=True,
        tiles=basemap,
        attr="Map tiles",  # Attribution text
        prefer_canvas=True,  # This might help with rendering
        basemap=basemap
    )

# Try disabling smooth zoom which can cause blur during transitions
    m.options['smoothZoom'] = False

    cluster = MarkerCluster().add_to(m)
    
    
    for _, row in places.iterrows():
        folium.Marker(
            location=[row['latitude'], row['longitude']],
            popup=folium.Popup(f"{row['token']}", parse_html=True),
            icon=folium.Icon(color="red"),
            tooltip=f"Modern {row['name']}, count {row['frekv']}"
        ).add_to(cluster)

    folium.LayerControl().add_to(m)
    m.to_streamlit(height=600)
    