"""
Streamlit-based GUI for the Etymologistics application.

This module implements a clean, minimal user interface for etymology exploration.
The design emphasizes readability and intuitive data presentation while
maintaining extensibility for future features.
"""

import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from loguru import logger
import asyncio
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from typing import List, Tuple
from src.rag.rag_pipeline import RAGPipeline

# Configure page and state
st.set_page_config(
    page_title="Etymologistics",
    page_icon="🌳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for clean typography and spacing
st.markdown("""
<style>
    /* Global Styles */
    .main {
        max-width: 1200px;
        padding: 2rem;
    }
    .stButton button {
        width: 100%;
        border-radius: 4px;
        padding: 0.5rem;
    }
    
    /* Etymology Tree Styles */
    .etymology-container {
        display: flex;
        gap: 2rem;
        margin: 1rem 0;
    }
    
    .etymology-tree {
        flex: 2;
        padding: 1.5rem;
        background: linear-gradient(135deg, #1a1c1f 0%, #22262c 100%);
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        min-height: 600px;
        position: relative;
        overflow: hidden;
    }
    
    /* Node Styles with Animations */
    .tree-node {
        padding: 1rem;
        margin: 0.75rem 0;
        background: rgba(255, 255, 255, 0.04);
        border-radius: 8px;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        cursor: pointer;
        position: relative;
        overflow: hidden;
    }
    
    .tree-node:hover {
        background: rgba(255, 255, 255, 0.08);
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
    }
    
    .tree-node::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: linear-gradient(45deg, transparent, rgba(255, 255, 255, 0.1), transparent);
        transform: translateX(-100%);
        transition: transform 0.6s;
    }
    
    .tree-node:hover::before {
        transform: translateX(100%);
    }
    
    /* Node Type Styles */
    .node-root { 
        border-left: 4px solid #e34c26;
        background: linear-gradient(90deg, rgba(227, 76, 38, 0.1), transparent);
    }
    .node-intermediate { 
        border-left: 4px solid #58a6ff;
        background: linear-gradient(90deg, rgba(88, 166, 255, 0.1), transparent);
    }
    .node-modern { 
        border-left: 4px solid #3fb950;
        background: linear-gradient(90deg, rgba(63, 185, 80, 0.1), transparent);
    }
    .node-uncertain { 
        border-left: 4px solid #f0883e;
        background: linear-gradient(90deg, rgba(240, 136, 62, 0.1), transparent);
    }
    .node-related {
        border-left: 4px solid #ff69b4;
        background: linear-gradient(90deg, rgba(255, 105, 180, 0.1), transparent);
    }
    .node-parallel {
        border-left: 4px solid #ffa500;
        background: linear-gradient(90deg, rgba(255, 165, 0, 0.1), transparent);
    }
    
    /* Interactive Controls */
    .interactive-controls {
        display: flex;
        gap: 1rem;
        margin: 1rem 0;
        padding: 1rem;
        background: rgba(255, 255, 255, 0.02);
        border-radius: 8px;
        backdrop-filter: blur(10px);
    }
    
    /* Tooltip Styles */
    .tooltip {
        position: absolute;
        background: rgba(0, 0, 0, 0.9);
        color: white;
        padding: 0.5rem;
        border-radius: 4px;
        font-size: 0.875rem;
        z-index: 1000;
        pointer-events: none;
        opacity: 0;
        transition: opacity 0.2s;
    }
    
    .tree-node:hover .tooltip {
        opacity: 1;
    }
    
    /* Legend Styles */
    .tree-legend {
        position: absolute;
        bottom: 1rem;
        right: 1rem;
        padding: 1rem;
        background: rgba(0, 0, 0, 0.8);
        border-radius: 8px;
        backdrop-filter: blur(10px);
        z-index: 100;
        transition: opacity 0.3s;
    }
    
    .legend-item {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        margin: 0.25rem 0;
        opacity: 0.7;
        transition: opacity 0.2s;
    }
    
    .legend-item:hover {
        opacity: 1;
    }
    
    .legend-color {
        width: 12px;
        height: 12px;
        border-radius: 50%;
        transition: transform 0.2s;
    }
    
    .legend-item:hover .legend-color {
        transform: scale(1.2);
    }
    
    /* Mini-map Styles */
    .tree-minimap {
        position: absolute;
        top: 1rem;
        right: 1rem;
        width: 150px;
        height: 150px;
        background: rgba(0, 0, 0, 0.8);
        border-radius: 8px;
        overflow: hidden;
        z-index: 100;
    }
    
    /* Node Details Panel */
    .node-details {
        position: fixed;
        top: 50%;
        right: -300px;
        transform: translateY(-50%);
        width: 300px;
        background: rgba(0, 0, 0, 0.9);
        border-radius: 8px 0 0 8px;
        padding: 1rem;
        transition: right 0.3s;
        z-index: 1000;
    }
    
    .node-details.active {
        right: 0;
    }
</style>
""", unsafe_allow_html=True)


class EtymologyUI:
    """
    Main UI controller for the etymology application.
    
    This class manages the UI state, component rendering, and data flow
    between the interface and the agent pipeline.
    """
    
    def __init__(self, pipeline):
        """
        Initialize UI state and agent pipeline.
        
        Args:
            pipeline: The etymology pipeline instance to use
        """
        self.pipeline = pipeline
        
        # Initialize session state
        if 'search_history' not in st.session_state:
            st.session_state.search_history = []
        if 'current_word' not in st.session_state:
            st.session_state.current_word = None
        if 'tree_data' not in st.session_state:
            st.session_state.tree_data = None
        if 'selected_node' not in st.session_state:
            st.session_state.selected_node = None
        if 'view_mode' not in st.session_state:
            st.session_state.view_mode = 'tree'
    
    def run(self):
        """Run the Streamlit application."""
        self._render_sidebar()
        
        # Main content area
        st.title("Etymologistics")
        st.markdown("""
        Explore word origins and relationships through etymology.
        Enter a word below to discover its linguistic history.
        """)
        
        # Search input
        col1, col2 = st.columns([4, 1])
        with col1:
            word = st.text_input(
                "Enter a word",
                key="word_input",
                placeholder="e.g., 'heart', 'wisdom', 'star'",
                help="Type a word and press Enter or click 'Research'"
            )
        with col2:
            search_button = st.button(
                "Research",
                key="search_button",
                help="Click to research the word's etymology"
            )
        
        # Process search
        if search_button and word:
            self._process_search(word)
        
        # Display results if available
        if st.session_state.current_word:
            # Add view selector
            view_options = ["Tree", "Timeline", "Stem Breakdown"]
            selected_view = st.radio(
                "View Mode",
                options=view_options,
                horizontal=True,
                key="view_selector"
            )
            
            if selected_view == "Tree":
                self._render_tree_view()
            elif selected_view == "Timeline":
                self._render_timeline_view()
            else:
                self._render_stem_breakdown()
    
    def _render_sidebar(self):
        """Render sidebar with history and settings."""
        with st.sidebar:
            st.subheader("Search History")
            if st.session_state.search_history:
                for hist_word in st.session_state.search_history:
                    if st.button(
                        f"📜 {hist_word}",
                        key=f"history_{hist_word}",
                        help=f"Click to view etymology for '{hist_word}'"
                    ):
                        self._process_search(hist_word)
            else:
                st.info("No searches yet. Try researching a word!")
            
            st.subheader("View Settings")
            view_mode = st.radio(
                "View Mode",
                options=["Tree", "Timeline"],
                index=0 if st.session_state.view_mode == "tree" else 1,
                key="view_mode_radio",
                help="Choose how to visualize the etymology"
            )
            st.session_state.view_mode = view_mode.lower()
            
            if st.session_state.view_mode == "tree":
                st.subheader("Tree Settings")
                st.slider(
                    "Zoom Level",
                    min_value=0.5,
                    max_value=2.0,
                    value=1.0,
                    step=0.1,
                    key="zoom_level",
                    help="Adjust tree visualization size"
                )
                st.selectbox(
                    "Layout",
                    options=["Hierarchical", "Radial", "Horizontal"],
                    index=0,
                    key="tree_layout",
                    help="Choose tree visualization style"
                )
                st.slider(
                    "Minimum confidence",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.5,
                    step=0.1,
                    key="min_confidence",
                    help="Filter nodes by confidence score"
                )
    
    def _process_search(self, word: str):
        """
        Process a word search and update the UI state.
        
        Args:
            word: The word to research
        """
        try:
            with st.spinner(f"Researching etymology for '{word}'..."):
                # Update search history
                if word not in st.session_state.search_history:
                    st.session_state.search_history.insert(0, word)
                    if len(st.session_state.search_history) > 10:
                        st.session_state.search_history.pop()
                
                # Create a new event loop in a separate thread for async operations
                def run_async(coro):
                    with ThreadPoolExecutor() as executor:
                        future = executor.submit(lambda: asyncio.run(coro))
                        return future.result()
                
                # Get etymology data with concurrent processing
                try:
                    etymology_result = run_async(self.pipeline.process_word(word))
                    
                    if etymology_result:
                        # Update state with all results
                        st.session_state.current_word = word
                        st.session_state.etymology_data = etymology_result.etymology
                        st.session_state.tree_data = etymology_result.tree.to_dict()
                        st.session_state.similarities = etymology_result.similarities
                        st.session_state.stem_analysis = etymology_result.stem_analysis
                        st.session_state.selected_node = None
                        st.session_state.show_results = True
                        
                        # Use standard rerun instead of experimental
                        st.rerun()
                    else:
                        logger.warning(f"No etymology found for {word}")
                        st.error("No etymology information found for this word.")
                        
                except Exception as e:
                    logger.error(f"Pipeline error for {word}: {str(e)}")
                    st.error(f"Error processing '{word}': {str(e)}")
                    
        except Exception as e:
            logger.error(f"Search error for {word}: {str(e)}")
            st.error(f"Error researching '{word}': {str(e)}")
            self.current_result = None
    
    def _render_results(self):
        """Render etymology results based on view mode."""
        if not st.session_state.get('etymology_data'):
            return

        # Determine which view to show based on mode
        if st.session_state.view_mode == "stem":
            self._render_stem_analysis()
        elif st.session_state.view_mode == "timeline":
            self._render_timeline_view()
        else:  # Default to tree view
            self._render_tree_view()

    def _render_stem_breakdown(self):
        """Render morphological stem breakdown."""
        try:
            if not hasattr(st.session_state, 'stem_analysis') or not st.session_state.stem_analysis:
                st.info("No stem analysis available yet. Enter a word to analyze its morphological structure.")
                return
            
            st.subheader("Morphological Analysis")
            
            # Display overall confidence score if available
            if hasattr(st.session_state.stem_analysis, 'confidence_score'):
                st.progress(
                    st.session_state.stem_analysis.confidence_score,
                    text=f"Analysis Confidence: {st.session_state.stem_analysis.confidence_score:.2f}"
                )
            
            # Check if we have stems to display
            if not hasattr(st.session_state.stem_analysis, 'stems') or not st.session_state.stem_analysis.stems:
                st.warning("No morphological stems were identified for this word.")
                return
            
            # Create tabs for different views
            stems_tab, etymology_tab, examples_tab = st.tabs(["Stems", "Etymology", "Examples"])
            
            # Sort stems by position if available
            sorted_stems = sorted(
                st.session_state.stem_analysis.stems,
                key=lambda s: s.position[0] if hasattr(s, 'position') and s.position else 0
            )
            
            with stems_tab:
                # Create a visual representation of the word split into stems
                word = st.session_state.stem_analysis.word
                st.markdown("### Word Structure")
                
                # Create columns for each stem
                stem_cols = st.columns(len(sorted_stems))
                for i, (stem, col) in enumerate(zip(sorted_stems, stem_cols)):
                    with col:
                        # Color code by stem type
                        colors = {
                            "prefix": "#58a6ff",
                            "root": "#3fb950",
                            "suffix": "#ff69b4"
                        }
                        color = colors.get(stem.stem_type.lower(), "#ffffff")
                        st.markdown(
                            f'<div style="text-align: center; padding: 10px; '
                            f'background: {color}20; border: 1px solid {color}; '
                            f'border-radius: 5px; margin: 2px;">'
                            f'<span style="font-size: 1.2em;">{stem.text}</span><br>'
                            f'<small>{stem.stem_type}</small>'
                            f'</div>',
                            unsafe_allow_html=True
                        )
                
                # Display detailed stem information
                for stem in sorted_stems:
                    with st.expander(f"{stem.text} ({stem.stem_type})", expanded=True):
                        cols = st.columns([2, 3])
                        
                        with cols[0]:
                            # Basic info and language details
                            st.markdown(f"**{stem.stem_type.title()} Morpheme**")
                            if isinstance(stem.language, dict):
                                st.markdown("**Language Details:**")
                                st.info(
                                    f"Origin: {stem.language.get('origin', 'Unknown')}\n\n"
                                    f"Family: {stem.language.get('family', 'Unknown')}\n\n"
                                    f"Period: {stem.language.get('period', 'Unknown')}"
                                )
                        
                        with cols[1]:
                            # Meaning and semantic fields
                            if isinstance(stem.meaning, dict):
                                st.markdown("**Core Meaning:**")
                                st.info(stem.meaning.get('core', 'Unknown'))
                                
                                if stem.meaning.get('extended'):
                                    st.markdown("**Extended Meanings:**")
                                    for meaning in stem.meaning['extended']:
                                        st.success(f"• {meaning}")
                                
                                if stem.meaning.get('semantic_fields'):
                                    st.markdown("**Semantic Fields:**")
                                    fields_cols = st.columns(min(3, len(stem.meaning['semantic_fields'])))
                                    for i, field in enumerate(stem.meaning['semantic_fields']):
                                        fields_cols[i % 3].markdown(f"🏷️ {field}")
            
            with etymology_tab:
                for stem in sorted_stems:
                    with st.expander(f"{stem.text} Development", expanded=True):
                        # Etymology details
                        if isinstance(stem.etymology, dict):
                            st.markdown("**Historical Development:**")
                            st.info(stem.etymology.get('development', 'Unknown'))
                            
                            if stem.etymology.get('cognates'):
                                st.markdown("**Cognates:**")
                                cognate_cols = st.columns(min(3, len(stem.etymology['cognates'])))
                                for i, cognate in enumerate(stem.etymology['cognates']):
                                    cognate_cols[i % 3].code(cognate)
                            
                            if stem.etymology.get('semantic_changes'):
                                st.markdown("**Semantic Changes:**")
                                for change in stem.etymology['semantic_changes']:
                                    st.markdown(f"↝ {change}")
                        
                        # Morphological features
                        if isinstance(stem.morphology, dict):
                            st.markdown("**Morphological Features:**")
                            
                            if stem.morphology.get('allomorphs'):
                                st.markdown("**Variant Forms:**")
                                st.code(" ~ ".join(stem.morphology['allomorphs']))
                            
                            if stem.morphology.get('combinations'):
                                st.markdown("**Common Combinations:**")
                                st.code(" + ".join(stem.morphology['combinations']))
                            
                            if stem.morphology.get('restrictions'):
                                st.markdown("**Usage Restrictions:**")
                                for restriction in stem.morphology['restrictions']:
                                    st.warning(restriction)
            
            with examples_tab:
                for stem in sorted_stems:
                    with st.expander(f"{stem.text} Usage", expanded=True):
                        if isinstance(stem.examples, dict):
                            cols = st.columns(2)
                            
                            with cols[0]:
                                if stem.examples.get('modern'):
                                    st.markdown("**Modern Usage**")
                                    for example in stem.examples['modern']:
                                        st.success(f"• {example}")
                                
                                if stem.examples.get('historical'):
                                    st.markdown("**Historical Usage**")
                                    for example in stem.examples['historical']:
                                        st.info(f"• {example}")
                            
                            with cols[1]:
                                if stem.examples.get('related_terms'):
                                    st.markdown("**Related Terms**")
                                    terms_cols = st.columns(min(3, len(stem.examples['related_terms'])))
                                    for i, term in enumerate(stem.examples['related_terms']):
                                        terms_cols[i % 3].code(term)
                                        
            # Add word formation pattern if available
            if hasattr(st.session_state.stem_analysis, 'word_formation_pattern'):
                st.markdown("### Word Formation Pattern")
                st.info(st.session_state.stem_analysis.word_formation_pattern)
            
            # Add morphological features if available
            if hasattr(st.session_state.stem_analysis, 'morphological_features'):
                st.markdown("### Morphological Features")
                features_cols = st.columns(min(3, len(st.session_state.stem_analysis.morphological_features)))
                for i, feature in enumerate(st.session_state.stem_analysis.morphological_features):
                    features_cols[i % 3].markdown(f"• {feature}")
                    
        except Exception as e:
            logger.error(f"Error rendering stem breakdown: {str(e)}")
            st.error("Error displaying stem analysis. Please try refreshing the page or try a different word.")
    
    def _render_tree_view(self):
        """Render the etymology tree visualization."""
        if 'etymology_data' not in st.session_state or 'tree_data' not in st.session_state:
            return

        st.header("Etymology Tree")
        
        # Show earliest ancestors
        etymology = st.session_state.etymology_data
        if hasattr(etymology, 'earliest_ancestors') and etymology.earliest_ancestors:
            st.subheader("Earliest Known Forms")
            for lang, word in etymology.earliest_ancestors:
                st.info(f"• {word} ({lang})")
        
        # Add etymology summary
        with st.expander("Etymology Summary", expanded=True):
            cols = st.columns(2)
            
            with cols[0]:
                if hasattr(etymology, 'confidence_score'):
                    st.progress(
                        etymology.confidence_score,
                        text=f"Research Confidence: {etymology.confidence_score:.2f}"
                    )
            
            with cols[1]:
                if hasattr(etymology, 'sources'):
                    st.markdown("**Sources:**")
                    for source in etymology.sources:
                        if hasattr(source, 'url') and source.url:
                            st.markdown(f"• [{source.name}]({source.url})")
                        else:
                            st.markdown(f"• {source.name}")
        
        # Create nodes and edges for visualization
        nodes = []
        edges = []
        
        def process_node(node, parent_id=None, level=0, index=0, total=1):
            """Process node and calculate its position in the tree."""
            node_id = node["id"]
            
            # Calculate x, y coordinates based on tree layout
            layout = st.session_state.tree_layout.lower()
            if layout == "hierarchical":
                x = index / max(total, 1)  # Normalize to 0-1
                y = -level  # Negative to go top to bottom
            elif layout == "radial":
                angle = (index / max(total, 1)) * 2 * 3.14159  # Convert to radians
                radius = level + 1
                x = radius * np.cos(angle)
                y = radius * np.sin(angle)
            else:  # horizontal
                y = index / max(total, 1)  # Normalize to 0-1
                x = -level  # Negative to go left to right
            
            # Add semantic changes and confidence to label
            confidence_indicator = "★" * int(node["confidence"] * 5)  # 0-5 stars
            label = [
                f"{node['word']}",
                f"{node['language']}",
                f"{confidence_indicator}"
            ]
            if node.get('semantic_changes'):
                label.append(f"↝ {', '.join(node['semantic_changes'])}")
            
            # Create hover text with detailed information
            hover_info = [
                f"Word: {node['word']}",
                f"Language: {node['language']}",
                f"Date: {node['date'] or 'Unknown'}",
                f"Confidence: {node['confidence']:.2f}",
            ]
            if node.get('notes'):
                hover_info.append(f"Notes: {node['notes']}")
            if node.get('semantic_changes'):
                hover_info.append(f"Semantic Changes: {', '.join(node['semantic_changes'])}")
            
            nodes.append({
                "id": node_id,
                "x": x,
                "y": y,
                "label": "<br>".join(label),
                "title": "<br>".join(hover_info),
                "color": self._get_node_color(node["style"]),
                "size": 30 * (0.5 + node["confidence"] * 0.5),  # Size varies less dramatically
                "symbol": self._get_node_symbol(node["style"]),
                "style": node["style"]
            })
            
            if parent_id:
                edges.append({
                    "from": parent_id,
                    "to": node_id,
                    "style": "evolution",
                    "width": 2 * node["confidence"]
                })
            
            # Process children
            child_count = len(node["children"])
            for i, child in enumerate(node["children"]):
                process_node(child, node_id, level + 1, i, child_count)
            
            # Process parallel nodes with curved connections
            parallel_count = len(node.get("parallel_nodes", []))
            for i, parallel in enumerate(node.get("parallel_nodes", [])):
                parallel_angle = (i / max(parallel_count, 1) - 0.5) * np.pi / 2
                parallel_radius = 0.5
                parallel_x = x + parallel_radius * np.cos(parallel_angle)
                parallel_y = y + parallel_radius * np.sin(parallel_angle)
                
                parallel_id = f"{parallel['id']}_parallel"
                nodes.append({
                    "id": parallel_id,
                    "x": parallel_x,
                    "y": parallel_y,
                    "label": f"{parallel['word']}<br>{parallel['language']}",
                    "title": f"Parallel Evolution<br>{parallel['notes'] or ''}",
                    "color": self._get_node_color("parallel"),
                    "size": 25 * parallel["confidence"],
                    "symbol": "diamond",
                    "style": "parallel"
                })
                
                # Create curved edge for parallel evolution
                edges.append({
                    "from": node_id,
                    "to": parallel_id,
                    "style": "parallel",
                    "curve": 0.3,
                    "width": 1.5 * parallel["confidence"]
                })
            
            # Process relations with curved connections
            for rel_type, related_nodes in node.get("relations", {}).items():
                for i, related in enumerate(related_nodes):
                    angle = (i / len(related_nodes)) * 2 * np.pi
                    radius = 0.7
                    related_x = x + radius * np.cos(angle)
                    related_y = y + radius * np.sin(angle)
                    
                    related_id = f"{related['id']}_{rel_type}"
                    nodes.append({
                        "id": related_id,
                        "x": related_x,
                        "y": related_y,
                        "label": f"{related['word']}<br>{related['language']}",
                        "title": f"{rel_type.title()} Relationship<br>{related['notes'] or ''}",
                        "color": self._get_node_color("related"),
                        "size": 20 * related["confidence"],
                        "symbol": self._get_relation_symbol(rel_type),
                        "style": "related"
                    })
                    
                    # Create curved edge for relationship
                    edges.append({
                        "from": node_id,
                        "to": related_id,
                        "style": rel_type,
                        "curve": -0.3,
                        "width": 1.5 * related["confidence"]
                    })
        
        # Start processing from root
        process_node(st.session_state.tree_data)
        
        # Create Plotly figure with improved styling
        fig = go.Figure()
        
        # Add edges with enhanced styles
        edge_styles = {
            "evolution": dict(color="rgba(255, 255, 255, 0.4)", width=2),
            "parallel": dict(color="rgba(255, 165, 0, 0.4)", width=1.5, dash="dot"),
            "cognate": dict(color="rgba(0, 255, 0, 0.3)", width=1.5, dash="dash"),
            "borrowing": dict(color="rgba(255, 0, 0, 0.3)", width=1.5, dash="dash"),
            "semantic": dict(color="rgba(0, 0, 255, 0.3)", width=1.5, dash="dot"),
            "compound": dict(color="rgba(255, 0, 255, 0.3)", width=1.5, dash="dash")
        }
        
        for edge in edges:
            start = next(n for n in nodes if n["id"] == edge["from"])
            end = next(n for n in nodes if n["id"] == edge["to"])
            style = edge_styles.get(edge["style"], edge_styles["evolution"]).copy()
            style["width"] = style["width"] * edge.get("width", 1)
            
            # Create curved path if specified
            if "curve" in edge:
                path = self._create_curved_path(
                    start["x"], start["y"],
                    end["x"], end["y"],
                    edge["curve"]
                )
                x, y = zip(*path)
            else:
                x = [start["x"], end["x"]]
                y = [start["y"], end["y"]]
            
            fig.add_trace(go.Scatter(
                x=x,
                y=y,
                mode="lines",
                line=style,
                hoverinfo="skip",
                showlegend=False
            ))
        
        # Add nodes with enhanced styling
        for node_style in ["root", "intermediate", "modern", "uncertain", "parallel", "related"]:
            style_nodes = [n for n in nodes if n["style"] == node_style]
            if not style_nodes:
                continue
            
            fig.add_trace(go.Scatter(
                x=[n["x"] for n in style_nodes],
                y=[n["y"] for n in style_nodes],
                mode="markers+text",
                marker=dict(
                    size=[n["size"] for n in style_nodes],
                    color=[n["color"] for n in style_nodes],
                    symbol=[n["symbol"] for n in style_nodes],
                    line=dict(
                        width=1,
                        color="rgba(255, 255, 255, 0.5)"
                    ),
                    gradient=dict(
                        type="radial",
                        color="white"
                    )
                ),
                text=[n["label"] for n in style_nodes],
                textposition="top center",
                hovertext=[n["title"] for n in style_nodes],
                hoverinfo="text",
                name=node_style.title(),
                hoverlabel=dict(
                    bgcolor="rgba(0,0,0,0.8)",
                    font_size=12,
                    font_family="monospace"
                )
            ))
        
        # Update layout with enhanced styling
        zoom = st.session_state.zoom_level
        fig.update_layout(
            showlegend=True,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01,
                bgcolor="rgba(0,0,0,0.5)",
                bordercolor="rgba(255,255,255,0.2)",
                borderwidth=1,
                font=dict(size=10)
            ),
            hovermode="closest",
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(
                showgrid=False,
                zeroline=False,
                showticklabels=False,
                range=[-2*zoom, 2*zoom]
            ),
            yaxis=dict(
                showgrid=False,
                zeroline=False,
                showticklabels=False,
                range=[-2*zoom, 2*zoom],
                scaleanchor="x",
                scaleratio=1
            ),
            margin=dict(l=0, r=0, t=0, b=0),
            height=600,
            dragmode="pan",
            modebar=dict(
                bgcolor="rgba(0,0,0,0.5)",
                color="white",
                activecolor="lightblue"
            ),
            updatemenus=[
                dict(
                    type="buttons",
                    showactive=False,
                    buttons=[
                        dict(
                            label="Reset View",
                            method="relayout",
                            args=[{"xaxis.range": [-2*zoom, 2*zoom],
                                  "yaxis.range": [-2*zoom, 2*zoom]}]
                        ),
                        dict(
                            label="Zoom In",
                            method="relayout",
                            args=[{"xaxis.range": [-zoom, zoom],
                                  "yaxis.range": [-zoom, zoom]}]
                        )
                    ],
                    pad={"r": 10, "t": 10},
                    x=0.1,
                    xanchor="left",
                    y=1.1,
                    yanchor="top",
                    bgcolor="rgba(0,0,0,0.5)",
                    font=dict(color="white")
                )
            ]
        )
        
        # Display visualization with enhanced interactivity
        st.plotly_chart(
            fig,
            use_container_width=True,
            config={
                "scrollZoom": True,
                "displayModeBar": True,
                "modeBarButtonsToAdd": ["drawclosedpath", "eraseshape"],
                "modeBarButtonsToRemove": ["select2d", "lasso2d"]
            }
        )
        
        # Display legend with enhanced styling
        self._render_tree_legend()
    
    def _render_tree_legend(self):
        """Render legend for tree visualization."""
        st.markdown("### Legend")
        
        # Create two columns for nodes and edges
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Node Types")
            node_types = [
                ("Earliest Known Form", "#e34c26"),
                ("Historical Form", "#58a6ff"),
                ("Modern Form", "#3fb950"),
                ("Uncertain Form", "#f0883e"),
                ("Related Word", "#ff69b4"),
                ("Parallel Evolution", "#ffa500")
            ]
            for label, color in node_types:
                st.markdown(
                    f'<div class="legend-item">'
                    f'<div class="legend-color" style="background: {color}"></div>'
                    f'<span>{label}</span>'
                    f'</div>',
                    unsafe_allow_html=True
                )
        
        with col2:
            st.markdown("#### Relationship Types")
            edge_types = [
                ("Direct Evolution", "solid", "#ffffff"),
                ("Parallel Development", "dotted", "#ffa500"),
                ("Cognate", "dashed", "#00ff00"),
                ("Borrowing", "dashed", "#ff0000"),
                ("Semantic", "dotted", "#0000ff"),
                ("Compound", "dashed", "#ff00ff")
            ]
            for label, style, color in edge_types:
                st.markdown(
                    f'<div class="legend-item">'
                    f'<div class="legend-line" style="border-top: 2px {style} {color}"></div>'
                    f'<span>{label}</span>'
                    f'</div>',
                    unsafe_allow_html=True
                )
    
    def _render_timeline_view(self):
        """Render timeline visualization."""
        st.subheader(f"Etymology Timeline for '{st.session_state.current_word}'")
        
        tree_data = st.session_state.tree_data
        if not tree_data:
            st.warning("No etymology data available.")
            return
        
        # Extract timeline data
        timeline_data = []
        
        def process_node(node):
            # Parse and normalize date
            date = node["date"]
            if date and isinstance(date, str):
                try:
                    # Try to parse numerical year
                    if date.isdigit() or (date.startswith('-') and date[1:].isdigit()):
                        date = int(date)
                    # Keep string dates (like "Unknown" or descriptive dates) as is
                except ValueError:
                    pass
            
            timeline_data.append({
                "word": node["word"],
                "language": node["language"],
                "date": date or "Unknown",
                "notes": node["notes"] or "",
                "confidence": node["confidence"],
                "style": node["style"]
            })
            for child in node["children"]:
                process_node(child)
        
        process_node(tree_data)
        
        # Create timeline visualization
        df = pd.DataFrame(timeline_data)
        
        # Sort by date, handling both numerical and string dates
        df['date_sort'] = pd.to_numeric(df['date'], errors='coerce')
        df = df.sort_values('date_sort', na_position='last')
        df = df.drop('date_sort', axis=1)
        
        fig = px.timeline(
            df,
            x_start="date",
            y="language",
            color="style",
            hover_data=["word", "notes", "confidence"],
            color_discrete_map={
                "root": "#e34c26",
                "intermediate": "#58a6ff",
                "modern": "#3fb950",
                "uncertain": "#f0883e"
            }
        )
        
        # Improve timeline layout
        fig.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            showlegend=True,
            height=400,
            xaxis=dict(
                type='category' if df['date'].dtype == 'object' else '-',
                title="Time Period",
                showgrid=True,
                gridcolor="rgba(255, 255, 255, 0.1)"
            ),
            yaxis=dict(
                title="Language",
                showgrid=True,
                gridcolor="rgba(255, 255, 255, 0.1)"
            )
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def _get_node_color(self, style: str) -> str:
        """Get color for node style."""
        colors = {
            "root": "#e34c26",
            "intermediate": "#58a6ff",
            "modern": "#3fb950",
            "uncertain": "#f0883e",
            "related": "#ff69b4",
            "parallel": "#ffa500"
        }
        return colors.get(style, "#ffffff")

    def _get_node_symbol(self, style: str) -> str:
        """Get symbol for node style."""
        symbols = {
            "root": "star",
            "intermediate": "circle",
            "modern": "square",
            "uncertain": "diamond",
            "related": "cross",
            "parallel": "diamond-tall"
        }
        return symbols.get(style, "circle")

    def _get_relation_symbol(self, relation_type: str) -> str:
        """Get symbol for relation type."""
        symbols = {
            "cognate": "star-triangle-up",
            "borrowing": "arrow-up",
            "semantic": "circle-cross",
            "compound": "square-cross"
        }
        return symbols.get(relation_type, "circle")

    def _create_curved_path(
        self,
        x0: float,
        y0: float,
        x1: float,
        y1: float,
        curvature: float,
        points: int = 50
    ) -> List[Tuple[float, float]]:
        """Create a curved path between two points."""
        # Calculate control point for quadratic Bezier curve
        mid_x = (x0 + x1) / 2
        mid_y = (y0 + y1) / 2
        normal_x = -(y1 - y0)
        normal_y = x1 - x0
        length = np.sqrt(normal_x**2 + normal_y**2)
        normal_x = normal_x / length if length > 0 else 0
        normal_y = normal_y / length if length > 0 else 0
        
        control_x = mid_x + curvature * normal_x
        control_y = mid_y + curvature * normal_y
        
        # Generate points along the curve
        t = np.linspace(0, 1, points)
        x = (1-t)**2 * x0 + 2*(1-t)*t * control_x + t**2 * x1
        y = (1-t)**2 * y0 + 2*(1-t)*t * control_y + t**2 * y1
        
        return list(zip(x, y))


if __name__ == "__main__":
    # Initialize pipeline
    from src.rag import RAGPipeline
    pipeline = RAGPipeline()
    
    # Create and run UI
    ui = EtymologyUI(pipeline=pipeline)
    ui.run() 