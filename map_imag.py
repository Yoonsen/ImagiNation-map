import streamlit as st
import pandas as pd
import numpy as np
import leafmap.foliumap as leafmap
from folium.plugins import MarkerCluster, HeatMap
import folium
import tools_imag as ti
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

    titles = st.multiselect('Tittel', 
                          options=list(set(subkorpus.Verk)),
                          key="title_select")

    if titles != []:
        subkorpus = subkorpus[subkorpus.Verk.isin(titles)]
    

with col2:
    "### Statistikk og steder"
    # Display corpus stats
    st.write(f"Antall bøker i korpus: {len(subkorpus)}, antall forfattere: {len(list(set(subkorpus.author)))}")
    #st.write(f"Antall forfattere: {len(list(set(subkorpus.author)))}")

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
    
    max_books = st.slider(
        "Maximum number of books to process", 
        min_value=100, max_value=5000, value=400, step=100,
        help="Lower numbers give faster response but less complete picture"
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
    significant_places = (places.groupby('name')
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
    
            
    max_places = st.slider("Maximum places to show", 
                         min_value=1, max_value=500, value=200, step=10)
    # Calculate dispersion
    total_books = len(selected_dhlabids)
    significant_places['dispersion'] = significant_places['dhlabid'].apply(lambda x: len(x) / total_books)
    significant_places['score'] = significant_places['frekv'] #* significant_places['dispersion']
    
    # Get top places
    significant_places = significant_places.nlargest(max_places, 'score')
    significant_places_clean = significant_places.dropna(subset=['latitude', 'longitude','feature_class'])

with col3:
    "### Liste over steder i korpuset"
    st.dataframe(places[[
        "token","name","frekv","feature_code","feature_class"
    ]],
                hide_index=True)
# Create tabs for different visualizations

"## Forskjellige kart - klikk på en fane for å vise"

tab1, tab2 = st.tabs(["Steder", "Varmekart"])

with tab1:
    col_controls, col_map = st.columns([1, 4])
    
    with col_controls:
        basemap = st.selectbox("Map style", BASEMAP_OPTIONS, 
                             index=BASEMAP_OPTIONS.index("OpenStreetMap.Mapnik"))

        
        clustering_zoom = st.slider("Disable clustering at zoom level", 
                                  min_value=1, max_value=15, value=8, step=1)
        
        marker_size = st.slider("Marker size multiplier", 
                              min_value=1, max_value=10, value=6, step=1)
    
    with col_map:
        significant_places_clean = significant_places.dropna(subset=['latitude', 'longitude'])
        center_lat = significant_places_clean['latitude'].median()
        center_lon = significant_places_clean['longitude'].median()
        
        m = leafmap.Map(
           center=(center_lat, center_lon),
           zoom=5,
           basemap=basemap
        )
  
        

            
        
        # Updated JavaScript function with blue colors for clusters
        cluster_js = """
            function(cluster) {
                var childCount = cluster.getChildCount();
                var total_freq = 0;
                
                cluster.getAllChildMarkers().forEach(function(marker) {
                    total_freq += marker.options.frequency;
                });
                
                var radius = Math.min(Math.sqrt(total_freq) * 2, 280);
                var borderWidth = Math.min(3 + Math.sqrt(childCount), 8);
                var innerRadius = Math.max(radius * 0.7, 20);
                
                return L.divIcon({
                    html: '<div style="display: flex; align-items: center; justify-content: center;">' +
                          '<div style="width: ' + radius * 2 + 'px; height: ' + radius * 2 + 'px; ' +
                          'background-color: rgba(0, 0, 255, 0.2); ' +  // Changed to blue
                          'border: ' + borderWidth + 'px solid rgba(0, 0, 255, 0.6); ' +  // Changed to blue
                          'border-radius: 50%; display: flex; align-items: center; justify-content: center;">' +
                          '<div style="width: ' + innerRadius + 'px; height: ' + innerRadius + 'px; ' +
                          'background-color: rgba(0, 0, 255, 0.4); border-radius: 50%; ' +  // Changed to blue
                          'display: flex; align-items: center; justify-content: center;">' +
                          '<span style="color: white; font-weight: bold;">' + childCount + '</span>' +
                          '</div></div></div>',
                    className: 'marker-cluster-custom',
                    iconSize: L.point(radius * 2, radius * 2),
                    iconAnchor: L.point(radius, radius)
                });
            }
        """
        
        cluster = MarkerCluster(
            icon_create_function=cluster_js,
            options={
                'maxClusterRadius': 80,  # Reduced from 160 to create smaller clusters
                'spiderfyOnMaxZoom': True,
                'showCoverageOnHover': True,
                'zoomToBoundsOnClick': True,
                'spiderfyDistanceMultiplier': 3,  # Increased from 2 to spread markers further
                'disableClusteringAtZoom': clustering_zoom,  # Added to stop clustering at higher zoom levels
                'animateAddingMarkers': True,
                'spiderLegPolylineOptions': {'weight': 1.5, 'color': '#222', 'opacity': 0.5},
                'zoomToBoundsOnClick': True,
                'singleMarkerMode': False,
                'spiderfyDistanceMultiplier': 3,
            }
        ).add_to(m)

        # First, create a mapping for feature classes and their colors
        feature_colors = {
            'P': 'red',      # Populated places
            'H': 'blue',     # Hydrographic
            'T': 'green',    # Mountain,hill
            'L': 'orange',   # Parks,areas
            'A': 'purple',   # Administrative
            'R': 'darkred',  # Roads,railroads
            'S': 'darkblue', # Spots,buildings,farms
            'V': 'darkgreen' # Forest,heath
        }
        
        # Add feature descriptions
        feature_descriptions = {
            'P': 'Befolkede steder',     
            'H': 'Vann og vassdrag',     
            'T': 'Fjell og høyder',      
            'L': 'Parker og områder',    
            'A': 'Administrative',       
            'R': 'Veier og jernbane',    
            'S': 'Bygninger og gårder',  
            'V': 'Skog og mark'          
        }
        
        feature_clusters = {}
        for feature_class in significant_places['feature_class'].unique():
            # Only use the description for the layer name, not the code
            layer_name = feature_descriptions.get(feature_class, f'Andre')  # Just the description
            feature_clusters[feature_class] = MarkerCluster(
                name=layer_name,  # This will show in the layer control
                icon_create_function=cluster_js,
                options={
                    'maxClusterRadius': 80,
                    'spiderfyOnMaxZoom': True,
                    'showCoverageOnHover': True,
                    'zoomToBoundsOnClick': True,
                    'spiderfyDistanceMultiplier': 3,
                    'disableClusteringAtZoom': clustering_zoom,
                    'animateAddingMarkers': True,
                    'spiderLegPolylineOptions': {'weight': 1.5, 'color': '#222', 'opacity': 0.5},
                }
            ).add_to(m)
            
 
            
        # Individual markers remain red
        # Add markers to their respective feature clusters
        for _, place in significant_places.iterrows():
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
            
            color = feature_colors.get(place['feature_class'], 'gray')
    
            radius = min(np.sqrt(place['frekv']) * marker_size, 80)
            marker = folium.CircleMarker(
                radius=radius,
                location=[place['latitude'], place['longitude']],
                popup=folium.Popup(html, max_width=500),
                tooltip=f"{place['name']} ({feature_descriptions.get(place['feature_class'], 'Annet')}): {place['frekv']} mentions in {book_count} books",
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=0.4,
                weight=2,
                frequency=float(place['frekv'])
            )
    
            # Add to appropriate feature cluster instead of main cluster
            if place['feature_class'] in feature_clusters:
                feature_clusters[place['feature_class']].add_child(marker)
        
        folium.LayerControl().add_to(m)
        m.to_streamlit(height=700)

with tab2:
    col_controls, col_map = st.columns([1, 4])
    
    with col_controls:
        st.write("Heatmap Controls")
        heat_radius = st.slider("Heat radius", 
                              min_value=5, max_value=50, value=15, step=5)
        
        heat_blur = st.slider("Heat blur", 
                            min_value=5, max_value=50, value=15, step=5)
        
        max_intensity = st.slider("Max intensity", 
                                min_value=0.1, max_value=1.0, value=0.5, step=0.1)
    
    with col_map:
        # Create heatmap visualization
        m_heat = leafmap.Map(
            center=(places['latitude'].median(), places['longitude'].median()),
            zoom=5,
            basemap=basemap
        )
        
        # Convert places to heatmap data
        heat_data = [[row['latitude'], row['longitude'], row['frekv']] 
                    for _, row in places.iterrows()]
        
        # Add heatmap layer
        HeatMap(
            heat_data,
            radius=heat_radius,
            blur=heat_blur,
            max_zoom=1,
            max_val=max_intensity,
        ).add_to(m_heat)
        
        m_heat.to_streamlit(height=700)