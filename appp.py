import streamlit as st
import pandas as pd
import networkx as nx
import folium
import random
from itertools import permutations
import streamlit.components.v1 as components

def generate_graph(fc_data):
    num_nodes = len(fc_data)
    max_routes = 2
    required_flow = 1000  # Total flow across the graph

    # Extract city and coordinate data from the DataFrame
    city_data = {
        'city': fc_data['city'].tolist(),
        'lat': fc_data['latitude'].tolist(),
        'lng': fc_data['longitude'].tolist()
    }

    # Create a directed graph
    G = nx.DiGraph()

    # Add nodes to the graph with demand attributes
    total_demand = 0
    for i, city in enumerate(city_data['city']):
        if city != 'Mumbai':
            demand = random.randint(-required_flow // 2, required_flow // 2)
        else:
            demand = -required_flow
        G.add_node(city, demand=demand)
        total_demand += demand

    # Balance the total demand to sum to 0
    G.nodes[city_data['city'][-1]]['demand'] -= total_demand

    # Generate random edges between nodes
    for i in range(len(city_data['city'])):
        for j in range(i + 1, len(city_data['city'])):
            if random.random() < 0.5:
                num_routes = random.randint(1, max_routes)
                for _ in range(num_routes):
                    distance = random.randint(50, 500)  # Random distance between 50 to 500 km
                    cost = random.randint(10, 100)  # Random cost between 10 to 100 Euros
                    capacity = random.randint(100, 500)  # Random capacity between 100 to 500 units
                    G.add_edge(city_data['city'][i], city_data['city'][j], distance=distance, cost=cost, capacity=capacity)
                    G.add_edge(city_data['city'][j], city_data['city'][i], distance=distance, cost=cost, capacity=capacity)  # Reverse direction

    # Solve the Minimum Cost Flow problem
    flow_cost, flow_dict = nx.network_simplex(G)
    
    return G, flow_dict, city_data

def tsp_bruteforce(graph, cities, start_node):
    if start_node not in cities:
        raise ValueError(f"Start node {start_node} not in the list of cities.")

    # Create a subgraph with the selected cities
    subgraph = graph.subgraph(cities)

    min_path = None
    min_cost = float('inf')
    nodes = list(cities)
    nodes.remove(start_node)
    for perm in permutations(nodes):
        path = [start_node] + list(perm) + [start_node]
        cost = sum(subgraph[u][v]['cost'] for u, v in zip(path, path[1:]) if subgraph.has_edge(u, v))
        if cost < min_cost:
            min_cost = cost
            min_path = path
    return min_path, min_cost

def create_folium_map(city_data, optimal_route, G, flow_dict, fc_data):
    start_city = optimal_route[0]
    start_lat = city_data['lat'][city_data['city'].index(start_city)]
    start_lng = city_data['lng'][city_data['city'].index(start_city)]

    # Initialize the map
    m = folium.Map(location=[start_lat, start_lng], zoom_start=6)

    # Add markers for each city
    for city in city_data['city']:
        lat = city_data['lat'][city_data['city'].index(city)]
        lng = city_data['lng'][city_data['city'].index(city)]
        is_fc = city in fc_data['city'].tolist()  # Check if the city is an FC

        folium.Marker(
            location=[lat, lng],
            popup=folium.Popup(f"{city}<br>Distance: {G.nodes[city].get('demand', 'N/A')} km<br>Cost: {G.nodes[city].get('cost', 'N/A')} EUR<br>Capacity: {G.nodes[city].get('capacity', 'N/A')} units", max_width=300),
            icon=folium.Icon(color='blue' if is_fc else 'green')
        ).add_to(m)

    # Draw routes with arrows
    for i in range(len(optimal_route) - 1):
        start = optimal_route[i]
        end = optimal_route[i + 1]
        start_lat = city_data['lat'][city_data['city'].index(start)]
        start_lng = city_data['lng'][city_data['city'].index(start)]
        end_lat = city_data['lat'][city_data['city'].index(end)]
        end_lng = city_data['lng'][city_data['city'].index(end)]

        folium.PolyLine(
            locations=[[start_lat, start_lng], [end_lat, end_lng]],
            color='red',
            weight=2.5,
            opacity=0.8
        ).add_to(m)

        # Add an arrow
        folium.Marker(
            location=[end_lat, end_lng],
            icon=folium.DivIcon(html='<div style="font-size: 24px; color: red;">&#8595;</div>')
        ).add_to(m)

    # Save the map to an HTML file and return its path
    map_html = m._repr_html_()
    return map_html

def main():
    st.title("Optimal Route Finder")

    # Sidebar for inputs
    st.sidebar.header("Upload Files")
    fc_file = st.sidebar.file_uploader("Upload FC CSV", type=["csv"])
    dc_file = st.sidebar.file_uploader("Upload DC CSV", type=["csv"])

    # Display the uploaded files and their columns
    if fc_file and dc_file:
        fc_data = pd.read_csv(fc_file)
        dc_data = pd.read_csv(dc_file)

        st.sidebar.write("FC Data:")
        st.sidebar.write(fc_data)
        st.sidebar.write("FC Columns:")
        st.sidebar.write(fc_data.columns)  # Print column names

        st.sidebar.write("DC Data:")
        st.sidebar.write(dc_data)
        st.sidebar.write("DC Columns:")
        st.sidebar.write(dc_data.columns)  # Print column names

        # Check if required columns exist
        if 'latitude' not in fc_data.columns or 'longitude' not in fc_data.columns:
            st.sidebar.error("FC CSV must contain 'latitude' and 'longitude' columns.")
            return

        if 'latitude' not in dc_data.columns or 'longitude' not in dc_data.columns:
            st.sidebar.error("DC CSV must contain 'latitude' and 'longitude' columns.")
            return

        # Select FC and DCs
        st.sidebar.header("Select FC and DCs")
        selected_fc = st.sidebar.selectbox('Select FC', options=fc_data['city'].tolist())
        selected_dcs = st.sidebar.multiselect('Select DCs', options=dc_data['city'].tolist())

        # Enable button only if an FC and at least one DC are selected
        if selected_fc and selected_dcs:
            if st.sidebar.button('Calculate Optimal Route'):
                G, flow_dict, city_data = generate_graph(fc_data)

                # Calculate the optimal route
                selected_cities = [selected_fc] + selected_dcs
                optimal_route, optimal_cost = tsp_bruteforce(G, selected_cities, selected_fc)

                st.sidebar.write(f"Optimal Route: {optimal_route}")
                st.sidebar.write(f"Optimal Cost: {optimal_cost}")

                # Create and display the Folium map
                folium_map_html = create_folium_map(city_data, optimal_route, G, flow_dict, fc_data)
                components.html(folium_map_html, height=1000)
        else:
            st.sidebar.info("Please select an FC and at least one DC to enable the calculation.")

if __name__ == "__main__":
    main()
