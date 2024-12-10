def setup_area_selection_menu(m, geocoded_places, books_df):
    m.add_draw_control(export=True)
    
    def get_books(bounds, mode='within'):
        min_lon, min_lat, max_lon, max_lat = bounds
        places_in_area = set(place for place, data in geocoded_places.items()
                           if min_lon <= data['lon'] <= max_lon 
                           and min_lat <= data['lat'] <= max_lat)
        
        if mode == 'within':
            return [idx for idx, row in books_df.iterrows() 
                    if places_in_area.intersection(row['places'])]
        elif mode == 'exclude':
            return [idx for idx, row in books_df.iterrows() 
                    if not places_in_area.intersection(row['places'])]
        elif mode == 'exclusively':
            return [idx for idx, row in books_df.iterrows() 
                    if all(p in places_in_area for p in row['places'])]
    
    # Add context menu
    m.add_context_menu({
        'Books mentioning places in area': lambda bounds: get_books(bounds, 'within'),
        'Books without places in area': lambda bounds: get_books(bounds, 'exclude'),
        'Books exclusively in area': lambda bounds: get_books(bounds, 'exclusively')
    })
    
    return m

def setup_realtime_area_selection(m, geocoded_places, books_df):
    m.add_draw_control(export=True)
    
    def update_selection(bounds):
        # Convert current bounds to box
        min_lon, min_lat, max_lon, max_lat = bounds
        
        # Real-time filtering
        selected_books = []
        places_in_view = set(place for place, data in geocoded_places.items()
                           if min_lon <= data['lon'] <= max_lon 
                           and min_lat <= data['lat'] <= max_lat)
        
        return [idx for idx, row in books_df.iterrows() 
                if places_in_view.intersection(row['places'])]
    
    # Add mousemove event listener during drawing
    m.on_draw_vertex(update_selection)
    
    return m

def calculate_dispersion(books_df):
    """Calculate frequency and dispersion metrics for places"""
    total_books = len(books_df)
    place_stats = {}
    
    for book_id, places in enumerate(books_df['places']):
        for place in set(places):  # Count unique appearances per book
            if place not in place_stats:
                place_stats[place] = {'freq': 0, 'docs': set()}
            place_stats[place]['freq'] += places.count(place)
            place_stats[place]['docs'].add(book_id)
    
    # Calculate metrics
    for place in place_stats:
        stats = place_stats[place]
        stats['dispersion'] = len(stats['docs']) / total_books
        stats['score'] = stats['freq'] * stats['dispersion']  # Combined metric
    
    return place_stats

def select_places(place_stats, min_score=0.01, max_places=2000):
    """Select places based on combined frequency/dispersion score"""
    filtered = {p: s for p, s in place_stats.items() if s['score'] >= min_score}
    return dict(sorted(filtered.items(), 
                      key=lambda x: x[1]['score'], 
                      reverse=True)[:max_places])