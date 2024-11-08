import logging
import re
from docutils import nodes
from sphinx.transforms import SphinxTransform

logger = logging.getLogger(__name__)

# Mapping from alert types to docutils nodes
ALERT_MAP = {
    "NOTE": nodes.note,
    "TIP": nodes.tip,
    "IMPORTANT": nodes.important,
    "WARNING": nodes.warning,
    "CAUTION": nodes.caution,
}

class MarkdownAlertTransform(SphinxTransform):
    default_priority = 500

    def apply(self):
        # Get the list of affected documents from the Sphinx configuration
        affected_docs = self.env.config.md_alert_to_admonition_affected_docs
        if self.env.docname not in affected_docs:
            return

        logger.info(f"Searching for GitHub Alert blocks in: {self.env.docname}")
        # Update the regex pattern to capture alert type (e.g., NOTE, TIP, IMPORTANT)
        search_pattern = re.compile(r'^\s{0,4}\[!(NOTE|TIP|IMPORTANT|WARNING|CAUTION)\]\s*(\n\s*\n)?')

        # Traverse through all block quote nodes in the document
        for node in self.document.traverse(nodes.block_quote):
            match = search_pattern.match(node.astext())
            if match:
                alert_type = match.group(1)  # Extract the alert type (e.g., NOTE, TIP)
                self.replace_with_admonition(node, alert_type, search_pattern)

    def replace_with_admonition(self, node, alert_type, search_pattern):
        """
        Replace block quote nodes containing '[!ALERT_TYPE]' text
        with the corresponding admonition node.
        """
        admonition_class = ALERT_MAP.get(alert_type)
        if not admonition_class:
            logger.warning(f"Unrecognized alert type '[!{alert_type}]' in document {self.env.docname}")
            return

        admonition_node = admonition_class()
        is_replaced = False

        for child in node.children:
            # Check if the child node is a paragraph and contains the alert tag
            if isinstance(child, nodes.paragraph):
                text = child.astext()
                if not is_replaced and search_pattern.match(text):
                    # Create a new paragraph to avoid modifying the original node's children in-place
                    new_paragraph = nodes.paragraph()

                    # Remove the '[!ALERT_TYPE]' tag if the paragraph is plain text
                    if len(child.children) == 1 and isinstance(child.children[0], nodes.Text):
                        # Plain text paragraph; clean text directly
                        cleaned_text = re.sub(search_pattern, '', text, count=1)
                        logger.info(f"Found and replaced !{alert_type} alert block")
                        if cleaned_text:
                            new_paragraph += nodes.Text(cleaned_text)
                            logger.debug(f"Replaced text: {cleaned_text}")
                    else:
                        # Mixed content; preserve existing structure and clean only the first text node
                        for subchild in child.children:
                            if isinstance(subchild, nodes.Text) and not is_replaced:
                                cleaned_text = re.sub(search_pattern, '', subchild.astext(), count=1)
                                logger.info(
                                    f"Found and replaced !{alert_type} alert block in formated text")
                                if cleaned_text:
                                    new_paragraph += nodes.Text(cleaned_text)
                                    logger.debug(f"Replaced text: {cleaned_text}")
                                is_replaced = True
                            else:
                                # Add other nodes (like strong, emphasis) unmodified
                                new_paragraph += subchild

                    # Add the new paragraph to the admonition node
                    admonition_node += new_paragraph
                    is_replaced = True
                    continue  # Skip adding the original child node

            # Add unmodified child nodes to the admonition node
            admonition_node += child

        # Replace the original node with the new admonition node
        node.replace_self(admonition_node)


def setup(app):
    app.add_transform(MarkdownAlertTransform)
    # Add a new config value 'markdown_alert_affected_docs' to specify affected documents
    app.add_config_value('md_alert_to_admonition_affected_docs', ['README'], 'env')
