import streamlit as st
import pandas as pd
import numpy as np
import leafmap.foliumap as leafmap
from folium.plugins import MarkerCluster, HeatMap
import folium
import tools_imag as ti
from urllib.parse import quote
from map_func import make_map

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
    corpus['author'] = corpus['author'].fillna('').apply(lambda x: x.replace('/', ' '))
    authors = list(set(corpus.author))
    titles = list(set(corpus.title))
    categories = list(set(corpus.category))
    return corpus, authors, titles, categories

@st.cache_data
def load_exploded_places():
    df = pd.read_pickle('exploded_places.pkl')
    #st.write("Columns in df:", df.columns)
    return df

@st.cache_data
def st_make_map(significant_places, corpus_df, basemap, marker_size, center=None, zoom=None):
    return make_map(significant_places, corpus_df, basemap, marker_size, center=center, zoom=zoom)
    
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


def change_map_view(m, center, zoom):
    """Updates map center and zoom level"""
    m.set_center(center[0], center[1])
    m.set_zoom(zoom)
    return m


# Define available basemaps
BASEMAP_OPTIONS = [
    "OpenStreetMap.Mapnik",
    "CartoDB.Positron",
    "CartoDB.DarkMatter",
   
]
st.set_page_config(layout="wide")
st.title("ImagiNation")

# Load initial data

corpus_df, authorlist, titlelist, categorylist = corpus()

# Corpus Builder Section
#st.header("Bygg et korpus")
subkorpus = corpus_df.copy()
subkorpus["Verk"] = subkorpus.apply(lambda x: f"{x['title'] or 'Uten tittel'} av {x['author'] or 'Ingen'} ({x['year'] or 'n.d.'})", axis=1)

col1, col2, col3 = st.columns([1, 1, 1])

with col1:
    "### Metadata"

    years = st.slider("Periode", min_value=1814, max_value=1905, value=(1850, 1880), 
                     help="Velg perdiode")
    subkorpus = subkorpus[(years[0] <= subkorpus.year) & (subkorpus.year <= years[1])]
    
    cola, colb = st.columns([1,1])
    with cola:
    
        categories = st.multiselect("Kategori", 
                                  options=categorylist,
                                  key="category_select")
        
        if categories:
            subkorpus = subkorpus[subkorpus.category.isin(categories)]
            
        authors = st.multiselect('Forfatter', 
                               options=list(set(subkorpus.author)),
                               key="author_select")
        if authors:
            subkorpus = subkorpus[subkorpus.author.isin(authors)]
    with colb:
        titles = st.multiselect('Tittel', 
                              options=list(set(subkorpus.Verk)),
                              key="title_select")
    
        if titles != []:
            subkorpus = subkorpus[subkorpus.Verk.isin(titles)]
    
        preprocessed_places = load_exploded_places()

        all_place_names = sorted(preprocessed_places['name'].unique())
        selected_places = st.multiselect(
            'Velg bøker som inneholder steder', 
            options=all_place_names,
            help="Vis kun bøker som inneholder valgte steder"
        )
        
        if selected_places:
            place_books = preprocessed_places[
                preprocessed_places['name'].isin(selected_places)
            ]['docs'].unique()
            subkorpus = subkorpus[subkorpus.dhlabid.isin(place_books)]

with col2:
    "### Statistikk og steder"
    
    # Display corpus stats
    st.write(f"Antall bøker i korpus: {len(subkorpus)}, antall forfattere: {len(list(set(subkorpus.author)))}")
    #st.write(f"Antall forfattere: {len(list(set(subkorpus.author)))}")

    

# Create tabs for different visualizations

"## Faner med forskjellige kart"


WORLD_VIEW = {
    'center': [20, 0],  # Approximately centers the world map
    'zoom': 2
}

EUROPE_VIEW = {
    'center': [55, 15],  # Centers on Europe
    'zoom': 4
}


tab1, tab2 = st.tabs(["Steder", "Varmekart"])

with tab1:


    
    col_controls, col_map = st.columns([1, 4])
    
    with col_controls:
        col_world, col_europe = st.columns(2)
        with col_world:
            world_btn = st.button("🌎 World View")
        with col_europe:
            europe_btn = st.button("🗺️ Europe View")
        
        max_books = st.slider(
            "Maks antall bøker i visning", 
            min_value=100, max_value=5000, value=400, step=100,
            help="Jo færre bøker jo raskere visning"
        )
        # Sample books if needed
        if len(subkorpus) > max_books:
            selected_dhlabids = subkorpus.sample(max_books).dhlabid
            st.write(f"Gjør et utvalg på {max_books} bøker")
        else:
            selected_dhlabids = subkorpus.dhlabid
            st.write(f"Bruker alle {len(subkorpus)} bøkene")
    
        # Get and process places
        places = ti.geo_locations_corpus(selected_dhlabids)
        places = places[places['rank']==1]
        st.write(f"Antall steder: {len(places)}")
    
        # Group places to get statistics
        all_places = (places.groupby('name')
            .agg({
                'token': 'first',
                'frekv': 'sum',
                'latitude': 'first',
                'longitude': 'first',
                'feature_class':'first',
                'dhlabid': list
            })
            .reset_index()
        )
        
                
        max_places = st.slider("Maks antall steder å vise", 
                             min_value=1, max_value=500, value=200, step=10)
        # Calculate dispersion
        total_books = len(selected_dhlabids)
        
        all_places['dispersion'] = all_places['dhlabid'].apply(lambda x: len(x) / total_books)
        all_places['score'] = all_places['frekv'] #* significant_places['dispersion']
        
        # Get top places
        significant_places = all_places.nlargest(max_places, 'score')
        significant_places_clean = significant_places.dropna(subset=['latitude', 'longitude','feature_class'])


        
        basemap = st.selectbox("Underlagskart", BASEMAP_OPTIONS, 
                             index=BASEMAP_OPTIONS.index("OpenStreetMap.Mapnik"))

        
        # clustering_zoom = st.slider("Disable clustering at zoom level", 
        #                           min_value=1, max_value=15, value=8, step=1)
        
        marker_size = st.slider("Justeringsfaktor for sirkel", 
                              min_value=1, max_value=10, value=6, step=1)
        

    with col_map:
       # Set default center and zoom
        center = EUROPE_VIEW['center']
        zoom = EUROPE_VIEW['zoom']
        
        # Update based on button clicks

        if world_btn:
            center = WORLD_VIEW['center']
            zoom = WORLD_VIEW['zoom']
        elif europe_btn:
            center = EUROPE_VIEW['center']
            zoom = EUROPE_VIEW['zoom']
            
        m = st_make_map(significant_places, corpus_df, basemap, marker_size, center=center, zoom=zoom)
        m.to_streamlit(height=700)


    #     m = st_make_map(significant_places, corpus_df, basemap, marker_size)
        
    #     # Add JavaScript to handle view changes
    #     m.get_root().html.add_child(folium.Element("""
    #     <script>
    #     // Wait for the map to be fully loaded
    #     setTimeout(function() {
    #         // In leafmap, the map object is typically stored in a different way
    #         var maps = document.getElementsByClassName('leaflet-container');
    #         if (maps.length > 0) {
    #             var map = maps[0];
    #             var leafletMap = map.__leaflet_map__;  // This gets the actual Leaflet map object
                
    #             // Add click handlers for the buttons
    #             document.querySelector('[data-testid="baseButton-world_view1"]').onclick = function() {
    #                 if (leafletMap) {
    #                     leafletMap.setView([20, 0], 2);
    #                 }
    #             };
                
    #             document.querySelector('[data-testid="baseButton-europe_view1"]').onclick = function() {
    #                 if (leafletMap) {
    #                     leafletMap.setView([55, 15], 4);
    #                 }
    #             };
    #         }
    #     }, 1000);  // Give the map time to initialize
    #     </script>
    # """))
        
    #     m.to_streamlit(height=700)

with col3:
    f"### Liste over alle {len(all_places)} navngitte steder i korpuset"
    st.dataframe(all_places[[
        "token","name","frekv","feature_class"
    ]],
                hide_index=True)
    
with tab2:
    
    col_controls, col_map = st.columns([1, 4])

    with col_controls:
           
        # Add the view buttons
        col_world, col_europe = st.columns(2)
        with col_world:
            world_btn_heat = st.button("🌎 World View", key="world_heat")
        with col_europe:
            europe_btn_heat = st.button("🗺️ Europe View", key="europe_heat")
        st.write(f"Varmekartet benytter alle steder ({len(all_places)} i alt)  fra alle bøkene {len(subkorpus)}")
        st.write("Juster varmekart")
        
        # User controls for heatmap
        heat_radius = st.slider(
            "Radius", 
            min_value=5, max_value=50, value=15, step=5
        )
        heat_blur = st.slider(
            "Uskarphet", 
            min_value=5, max_value=50, value=15, step=5
        )
        max_intensity = st.slider(
            "Maks intensitet", 
            min_value=0.1, max_value=1.0, value=0.5, step=0.1
        )

    with col_map:
        # Create the base map centered on median coordinates
        center = EUROPE_VIEW['center']
        zoom = EUROPE_VIEW['zoom']
        
        # Update based on button clicks
        if world_btn_heat:
            center = WORLD_VIEW['center']
            zoom = WORLD_VIEW['zoom']
        elif europe_btn_heat:
            center = EUROPE_VIEW['center']
            zoom = EUROPE_VIEW['zoom']
            
        # m_heat = leafmap.Map(
        #     center=center,
        #     zoom=zoom,
        #     basemap=basemap
        # )
        # # Prepare data for heatmap
        # heat_data = [
        #     [row['latitude'], row['longitude'], row['frekv']] 
        #     for _, row in all_places.iterrows()
        # ]

        # # Add heatmap layer
        # HeatMap(
        #     data=heat_data,
        #     radius=heat_radius,
        #     blur=heat_blur,
        #     max_zoom=1,
        #     max_val=max_intensity,
        # ).add_to(m_heat)

        # # Render the heatmap in Streamlit
        # m_heat.to_streamlit(height=700)
        m_heat = leafmap.Map(
            center=center,
            zoom=zoom,
            basemap=basemap
        )
        
        # Add JavaScript for heatmap view changes
        m_heat.get_root().html.add_child(folium.Element("""
            <script>
            var heatmap = document.querySelector('#map');  // Adjust selector if needed
            
            document.querySelector('[data-testid="baseButton-world_view2"]').onclick = function() {
                heatmap.setView([20, 0], 2);
            };
            
            document.querySelector('[data-testid="baseButton-europe_view2"]').onclick = function() {
                heatmap.setView([55, 15], 4);
            };
            </script>
        """))
        heat_data = [
            [row['latitude'], row['longitude'], row['frekv']] 
            for _, row in all_places.iterrows()
        ]
        # Add heatmap layer and render
        HeatMap(
            data=heat_data,
            radius=heat_radius,
            blur=heat_blur,
            max_zoom=1,
            max_val=max_intensity,
        ).add_to(m_heat)
        
        m_heat.to_streamlit(height=700)
