"""
Scene Graph Builder
==================

Builds NetworkX graphs from detected objects and their relationships.
Clean, consistent implementation following CAPito architecture patterns.
"""

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Any, Optional, Tuple
import logging
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer


class SceneGraphBuilder:
    """
    Builds scene graphs from detected objects and their features.
    
    Creates NetworkX graphs with nodes representing objects and edges 
    representing spatial and semantic relationships.
    """
    
    def __init__(
        self, 
        similarity_model: str = "all-MiniLM-L6-v2",
        device: str = "cpu"
    ):
        """
        Initialize scene graph builder.
        
        Args:
            similarity_model: Sentence transformer model for semantic similarity
            device: Device to run similarity model on
        """
        self.device = device
        self.logger = logging.getLogger(__name__)
        
        try:
            self.similarity_model = SentenceTransformer(similarity_model, device=device)
            self.logger.info(f"Similarity model loaded: {similarity_model}")
        except Exception as e:
            self.logger.warning(f"Failed to load similarity model: {e}")
            self.similarity_model = None
    
    def build_graph(
        self,
        objects: List[Dict[str, Any]],
        image_size: Tuple[int, int],
        similarity_threshold: float = 0.6,
        distance_threshold: float = 0.3
    ) -> nx.Graph:
        """
        Build scene graph from detected objects.
        
        Args:
            objects: List of object dictionaries with features
            image_size: (height, width) of the image
            similarity_threshold: Minimum similarity for semantic edges
            distance_threshold: Maximum distance for spatial edges
            
        Returns:
            NetworkX graph representing the scene
        """
        G = nx.Graph()
        
        if not objects:
            return G
        
        # Add nodes for each object
        for i, obj in enumerate(objects):
            node_id = f"obj_{i}"
            
            # Prepare node features
            features = {
                'coordinates': self._get_normalized_coordinates(obj, image_size),
                'bbox': obj.get('bbox', [0, 0, 1, 1]),
                'class_name': obj.get('class_name', 'unknown'),
                'confidence': obj.get('confidence', 0.0),
                'caption': obj.get('caption', ''),
                'depth': obj.get('depth', 0.5),
                'area': self._calculate_area(obj.get('bbox', [0, 0, 1, 1])),
                'track_id': obj.get('track_id', -1)
            }
            
            # Add pose information for humans
            if obj.get('class_name') == 'person' and 'pose' in obj:
                features['pose'] = obj['pose']
            
            G.add_node(node_id, **features)
        
        # Add edges based on relationships
        node_list = list(G.nodes())
        for i in range(len(node_list)):
            for j in range(i + 1, len(node_list)):
                node1, node2 = node_list[i], node_list[j]
                
                # Calculate relationship score
                relationship_score = self._calculate_relationship_score(
                    G.nodes[node1], 
                    G.nodes[node2],
                    similarity_threshold,
                    distance_threshold
                )
                
                if relationship_score > 0.0:
                    G.add_edge(
                        node1, 
                        node2, 
                        weight=relationship_score,
                        relationship_caption=""  # Empty for now as requested
                    )
        
        self.logger.info(f"Built graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
        return G
    
    def _get_normalized_coordinates(
        self, 
        obj: Dict[str, Any], 
        image_size: Tuple[int, int]
    ) -> List[float]:
        """
        Get normalized coordinates [x1, y1, x2, y2, depth] for an object.
        
        Args:
            obj: Object dictionary
            image_size: (height, width) of image
            
        Returns:
            Normalized coordinates with depth
        """
        bbox = obj.get('bbox', [0, 0, 1, 1])
        depth = obj.get('depth', 0.5)
        
        h, w = image_size
        
        # Normalize bbox coordinates to [0, 1] range
        x1 = bbox[0] / w
        y1 = bbox[1] / h
        x2 = bbox[2] / w
        y2 = bbox[3] / h
        
        return [x1, y1, x2, y2, depth]
    
    def _calculate_area(self, bbox: List[float]) -> float:
        """Calculate normalized area of bounding box."""
        return (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
    
    def _calculate_relationship_score(
        self,
        node1_features: Dict[str, Any],
        node2_features: Dict[str, Any],
        similarity_threshold: float,
        distance_threshold: float
    ) -> float:
        """
        Calculate relationship score between two objects.
        
        Args:
            node1_features: Features of first object
            node2_features: Features of second object
            similarity_threshold: Minimum similarity threshold
            distance_threshold: Maximum distance threshold
            
        Returns:
            Combined relationship score [0, 1]
        """
        # Calculate spatial distance
        spatial_score = self._calculate_spatial_similarity(
            node1_features['coordinates'],
            node2_features['coordinates']
        )
        
        # Calculate semantic similarity
        semantic_score = self._calculate_semantic_similarity(
            node1_features.get('caption', node1_features['class_name']),
            node2_features.get('caption', node2_features['class_name'])
        )
        
        # Calculate class label similarity
        class_score = self._calculate_class_similarity(
            node1_features['class_name'],
            node2_features['class_name']
        )
        
        # Combine scores with weights
        combined_score = (
            0.4 * spatial_score +
            0.4 * semantic_score +
            0.2 * class_score
        )
        
        # Apply thresholds
        if spatial_score < distance_threshold and semantic_score < similarity_threshold:
            return 0.0
        
        return combined_score
    
    def _calculate_spatial_similarity(
        self,
        coords1: List[float],
        coords2: List[float]
    ) -> float:
        """
        Calculate spatial similarity between two objects.
        
        Args:
            coords1: [x1, y1, x2, y2, depth] for object 1
            coords2: [x1, y1, x2, y2, depth] for object 2
            
        Returns:
            Spatial similarity score [0, 1]
        """
        try:
            # Calculate center points
            center1 = [(coords1[0] + coords1[2]) / 2, (coords1[1] + coords1[3]) / 2]
            center2 = [(coords2[0] + coords2[2]) / 2, (coords2[1] + coords2[3]) / 2]
            
            # Calculate 2D distance
            distance_2d = np.sqrt(
                (center1[0] - center2[0]) ** 2 + 
                (center1[1] - center2[1]) ** 2
            )
            
            # Calculate depth difference
            depth_diff = abs(coords1[4] - coords2[4])
            
            # Combine 2D distance and depth difference
            total_distance = distance_2d + 0.5 * depth_diff
            
            # Convert distance to similarity (inverse relationship)
            similarity = 1.0 / (1.0 + total_distance)
            
            return similarity
            
        except Exception as e:
            self.logger.warning(f"Spatial similarity calculation failed: {e}")
            return 0.0
    
    def _calculate_semantic_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate semantic similarity between two text descriptions.
        
        Args:
            text1: First text description
            text2: Second text description
            
        Returns:
            Semantic similarity score [0, 1]
        """
        if not self.similarity_model or not text1 or not text2:
            return 0.0
        
        try:
            # Get embeddings
            embeddings = self.similarity_model.encode([text1, text2])
            
            # Calculate cosine similarity
            similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
            
            # Normalize to [0, 1] range
            return (similarity + 1) / 2
            
        except Exception as e:
            self.logger.warning(f"Semantic similarity calculation failed: {e}")
            return 0.0
    
    def _calculate_class_similarity(self, class1: str, class2: str) -> float:
        """Calculate similarity between object class names."""
        if class1 == class2:
            return 1.0
        
        # Simple semantic groupings
        similar_classes = {
            'person': ['human', 'man', 'woman', 'child'],
            'vehicle': ['car', 'truck', 'bus', 'bicycle', 'motorcycle'],
            'animal': ['cat', 'dog', 'bird', 'horse'],
            'furniture': ['chair', 'table', 'sofa', 'bed']
        }
        
        for group, classes in similar_classes.items():
            if class1 in classes and class2 in classes:
                return 0.7
        
        return 0.0
    
    def visualize_graph(
        self,
        graph: nx.Graph,
        save_path: Optional[str] = None,
        show_labels: bool = True,
        node_size: int = 1000,
        font_size: int = 10
    ) -> None:
        """
        Visualize the scene graph.
        
        Args:
            graph: NetworkX graph to visualize
            save_path: Path to save visualization (optional)
            show_labels: Whether to show node labels
            node_size: Size of nodes
            font_size: Font size for labels
        """
        if graph.number_of_nodes() == 0:
            self.logger.warning("Cannot visualize empty graph")
            return
        
        plt.figure(figsize=(12, 8))
        
        # Use spring layout for nice visualization
        pos = nx.spring_layout(graph, k=1, iterations=50)
        
        # Draw nodes
        node_colors = []
        node_labels = {}
        
        for node, data in graph.nodes(data=True):
            # Color by class
            class_name = data.get('class_name', 'unknown')
            if class_name == 'person':
                node_colors.append('lightblue')
            elif class_name in ['car', 'truck', 'bus']:
                node_colors.append('lightcoral')
            elif class_name in ['cat', 'dog']:
                node_colors.append('lightgreen')
            else:
                node_colors.append('lightgray')
            
            # Create label
            if show_labels:
                confidence = data.get('confidence', 0.0)
                node_labels[node] = f"{class_name}\n({confidence:.2f})"
        
        # Draw graph
        nx.draw_networkx_nodes(
            graph, pos, 
            node_color=node_colors, 
            node_size=node_size,
            alpha=0.8
        )
        
        nx.draw_networkx_edges(
            graph, pos, 
            alpha=0.6, 
            edge_color='gray',
            width=2
        )
        
        if show_labels:
            nx.draw_networkx_labels(
                graph, pos, 
                labels=node_labels,
                font_size=font_size
            )
        
        # Draw edge weights
        edge_labels = {}
        for u, v, data in graph.edges(data=True):
            weight = data.get('weight', 0.0)
            edge_labels[(u, v)] = f"{weight:.2f}"
        
        nx.draw_networkx_edge_labels(
            graph, pos,
            edge_labels=edge_labels,
            font_size=8,
            font_color='red'
        )
        
        plt.title("Scene Graph Visualization", fontsize=16)
        plt.axis('off')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Graph visualization saved to: {save_path}")
        
        plt.show()
    
    def export_graph(
        self,
        graph: nx.Graph,
        output_path: str,
        format: str = "json"
    ) -> None:
        """
        Export graph to file.
        
        Args:
            graph: NetworkX graph to export
            output_path: Path to save file
            format: Export format ('json', 'gexf', 'graphml')
        """
        try:
            if format.lower() == "json":
                # Convert to JSON-serializable format
                graph_data = nx.node_link_data(graph)
                import json
                with open(output_path, 'w') as f:
                    json.dump(graph_data, f, indent=2)
            elif format.lower() == "gexf":
                nx.write_gexf(graph, output_path)
            elif format.lower() == "graphml":
                nx.write_graphml(graph, output_path)
            else:
                raise ValueError(f"Unsupported format: {format}")
            
            self.logger.info(f"Graph exported to: {output_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to export graph: {e}")
    
    def cleanup(self):
        """Clean up model resources."""
        if hasattr(self, 'similarity_model') and self.similarity_model is not None:
            del self.similarity_model
