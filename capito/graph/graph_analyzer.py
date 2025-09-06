"""
Graph Analyzer
==============

Analysis utilities for scene graphs including metrics and insights.
"""

import networkx as nx
import numpy as np
from typing import Dict, List, Any, Tuple
import logging


class GraphAnalyzer:
    """
    Provides analysis capabilities for scene graphs.
    
    Includes graph metrics, clustering, and insights generation.
    """
    
    def __init__(self):
        """Initialize graph analyzer."""
        self.logger = logging.getLogger(__name__)
    
    def analyze_graph(self, graph: nx.Graph) -> Dict[str, Any]:
        """
        Perform comprehensive analysis of scene graph.
        
        Args:
            graph: NetworkX graph to analyze
            
        Returns:
            Dictionary containing analysis results
        """
        analysis = {
            'basic_metrics': self._calculate_basic_metrics(graph),
            'centrality_metrics': self._calculate_centrality_metrics(graph),
            'clustering_metrics': self._calculate_clustering_metrics(graph),
            'object_distribution': self._analyze_object_distribution(graph),
            'spatial_analysis': self._analyze_spatial_patterns(graph)
        }
        
        return analysis
    
    def _calculate_basic_metrics(self, graph: nx.Graph) -> Dict[str, Any]:
        """Calculate basic graph metrics."""
        try:
            metrics = {
                'num_nodes': graph.number_of_nodes(),
                'num_edges': graph.number_of_edges(),
                'density': nx.density(graph),
                'is_connected': nx.is_connected(graph),
                'num_components': nx.number_connected_components(graph)
            }
            
            if graph.number_of_nodes() > 0:
                degrees = [d for n, d in graph.degree()]
                metrics.update({
                    'avg_degree': np.mean(degrees),
                    'max_degree': np.max(degrees),
                    'min_degree': np.min(degrees)
                })
            
            return metrics
            
        except Exception as e:
            self.logger.warning(f"Basic metrics calculation failed: {e}")
            return {}
    
    def _calculate_centrality_metrics(self, graph: nx.Graph) -> Dict[str, Any]:
        """Calculate centrality metrics for nodes."""
        try:
            if graph.number_of_nodes() == 0:
                return {}
            
            # Calculate various centrality measures
            degree_centrality = nx.degree_centrality(graph)
            betweenness_centrality = nx.betweenness_centrality(graph)
            closeness_centrality = nx.closeness_centrality(graph)
            
            # Find most central nodes
            most_central_node = max(degree_centrality, key=degree_centrality.get)
            
            return {
                'degree_centrality': degree_centrality,
                'betweenness_centrality': betweenness_centrality,
                'closeness_centrality': closeness_centrality,
                'most_central_node': most_central_node,
                'most_central_object': graph.nodes[most_central_node].get('class_name', 'unknown')
            }
            
        except Exception as e:
            self.logger.warning(f"Centrality calculation failed: {e}")
            return {}
    
    def _calculate_clustering_metrics(self, graph: nx.Graph) -> Dict[str, Any]:
        """Calculate clustering coefficients."""
        try:
            if graph.number_of_nodes() == 0:
                return {}
            
            clustering = nx.clustering(graph)
            avg_clustering = nx.average_clustering(graph)
            
            return {
                'clustering_coefficients': clustering,
                'average_clustering': avg_clustering
            }
            
        except Exception as e:
            self.logger.warning(f"Clustering calculation failed: {e}")
            return {}
    
    def _analyze_object_distribution(self, graph: nx.Graph) -> Dict[str, Any]:
        """Analyze distribution of object types."""
        try:
            object_counts = {}
            confidence_stats = {}
            
            for node, data in graph.nodes(data=True):
                class_name = data.get('class_name', 'unknown')
                confidence = data.get('confidence', 0.0)
                
                # Count objects by class
                object_counts[class_name] = object_counts.get(class_name, 0) + 1
                
                # Track confidence statistics
                if class_name not in confidence_stats:
                    confidence_stats[class_name] = []
                confidence_stats[class_name].append(confidence)
            
            # Calculate average confidence per class
            avg_confidence = {}
            for class_name, confidences in confidence_stats.items():
                avg_confidence[class_name] = np.mean(confidences)
            
            return {
                'object_counts': object_counts,
                'average_confidence': avg_confidence,
                'total_unique_classes': len(object_counts),
                'most_common_class': max(object_counts, key=object_counts.get) if object_counts else None
            }
            
        except Exception as e:
            self.logger.warning(f"Object distribution analysis failed: {e}")
            return {}
    
    def _analyze_spatial_patterns(self, graph: nx.Graph) -> Dict[str, Any]:
        """Analyze spatial patterns in the scene."""
        try:
            if graph.number_of_nodes() == 0:
                return {}
            
            depths = []
            areas = []
            positions = []
            
            for node, data in graph.nodes(data=True):
                coords = data.get('coordinates', [0, 0, 1, 1, 0.5])
                depth = coords[4] if len(coords) > 4 else 0.5
                area = data.get('area', 0.0)
                
                depths.append(depth)
                areas.append(area)
                
                # Calculate center position
                center_x = (coords[0] + coords[2]) / 2
                center_y = (coords[1] + coords[3]) / 2
                positions.append([center_x, center_y])
            
            analysis = {
                'depth_distribution': {
                    'mean': np.mean(depths),
                    'std': np.std(depths),
                    'min': np.min(depths),
                    'max': np.max(depths)
                },
                'area_distribution': {
                    'mean': np.mean(areas),
                    'std': np.std(areas),
                    'min': np.min(areas),
                    'max': np.max(areas)
                }
            }
            
            # Analyze spatial clustering
            if len(positions) > 1:
                positions = np.array(positions)
                # Calculate pairwise distances
                distances = []
                for i in range(len(positions)):
                    for j in range(i + 1, len(positions)):
                        dist = np.linalg.norm(positions[i] - positions[j])
                        distances.append(dist)
                
                analysis['spatial_clustering'] = {
                    'avg_distance': np.mean(distances),
                    'min_distance': np.min(distances),
                    'max_distance': np.max(distances)
                }
            
            return analysis
            
        except Exception as e:
            self.logger.warning(f"Spatial analysis failed: {e}")
            return {}
    
    def find_object_clusters(
        self, 
        graph: nx.Graph, 
        similarity_threshold: float = 0.7
    ) -> List[List[str]]:
        """
        Find clusters of similar objects.
        
        Args:
            graph: Scene graph
            similarity_threshold: Minimum similarity for clustering
            
        Returns:
            List of object clusters (each cluster is a list of node IDs)
        """
        try:
            # Create subgraph with only high-similarity edges
            high_sim_edges = [
                (u, v) for u, v, d in graph.edges(data=True)
                if d.get('weight', 0) >= similarity_threshold
            ]
            
            if not high_sim_edges:
                return [[node] for node in graph.nodes()]
            
            subgraph = graph.edge_subgraph(high_sim_edges)
            
            # Find connected components as clusters
            clusters = list(nx.connected_components(subgraph))
            
            # Include isolated nodes as single-node clusters
            all_clustered_nodes = set()
            for cluster in clusters:
                all_clustered_nodes.update(cluster)
            
            isolated_nodes = set(graph.nodes()) - all_clustered_nodes
            for node in isolated_nodes:
                clusters.append({node})
            
            return [list(cluster) for cluster in clusters]
            
        except Exception as e:
            self.logger.warning(f"Clustering failed: {e}")
            return []
    
    def get_object_relationships(
        self, 
        graph: nx.Graph, 
        node_id: str
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Get all relationships for a specific object.
        
        Args:
            graph: Scene graph
            node_id: ID of the node to analyze
            
        Returns:
            Dictionary with relationship information
        """
        try:
            if node_id not in graph.nodes():
                return {}
            
            relationships = {
                'direct_connections': [],
                'spatial_neighbors': [],
                'semantic_neighbors': []
            }
            
            # Get direct connections
            for neighbor in graph.neighbors(node_id):
                edge_data = graph.edges[node_id, neighbor]
                neighbor_data = graph.nodes[neighbor]
                
                relationship_info = {
                    'neighbor_id': neighbor,
                    'neighbor_class': neighbor_data.get('class_name', 'unknown'),
                    'relationship_strength': edge_data.get('weight', 0.0),
                    'relationship_caption': edge_data.get('relationship_caption', '')
                }
                
                relationships['direct_connections'].append(relationship_info)
            
            # Sort by relationship strength
            relationships['direct_connections'].sort(
                key=lambda x: x['relationship_strength'], 
                reverse=True
            )
            
            return relationships
            
        except Exception as e:
            self.logger.warning(f"Relationship analysis failed: {e}")
            return {}
    
    def generate_scene_summary(self, graph: nx.Graph) -> str:
        """
        Generate a natural language summary of the scene.
        
        Args:
            graph: Scene graph to summarize
            
        Returns:
            Text summary of the scene
        """
        try:
            if graph.number_of_nodes() == 0:
                return "Empty scene with no detected objects."
            
            analysis = self.analyze_graph(graph)
            
            # Basic statistics
            num_objects = analysis['basic_metrics']['num_nodes']
            object_dist = analysis['object_distribution']
            
            summary_parts = []
            
            # Object count and types
            if num_objects == 1:
                summary_parts.append("The scene contains a single object")
            else:
                summary_parts.append(f"The scene contains {num_objects} objects")
            
            # Most common object types
            if 'object_counts' in object_dist:
                object_counts = object_dist['object_counts']
                if object_counts:
                    most_common = max(object_counts, key=object_counts.get)
                    count = object_counts[most_common]
                    
                    if count > 1:
                        summary_parts.append(f"with {count} {most_common}s being the most common")
                    else:
                        # List all object types
                        object_types = list(object_counts.keys())
                        if len(object_types) <= 3:
                            types_str = ", ".join(object_types)
                            summary_parts.append(f"including {types_str}")
            
            # Connectivity
            if analysis['basic_metrics']['is_connected']:
                summary_parts.append("All objects appear to be spatially or semantically related")
            else:
                num_components = analysis['basic_metrics']['num_components']
                if num_components > 1:
                    summary_parts.append(f"with {num_components} distinct groups of related objects")
            
            # Central object
            if 'most_central_object' in analysis.get('centrality_metrics', {}):
                central_obj = analysis['centrality_metrics']['most_central_object']
                summary_parts.append(f"The {central_obj} appears to be the central focus of the scene")
            
            summary = ". ".join(summary_parts) + "."
            return summary
            
        except Exception as e:
            self.logger.warning(f"Scene summary generation failed: {e}")
            return "Unable to generate scene summary."
