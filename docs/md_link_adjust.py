import logging
import re
from docutils import nodes
from sphinx.transforms import SphinxTransform
from sphinx.addnodes import pending_xref  # Import pending_xref from sphinx.addnodes

logger = logging.getLogger(__name__)

class LinkAdjustTransform(SphinxTransform):
    default_priority = 500

    def apply(self):
        # Get the list of affected documents from the Sphinx configuration
        affected_docs = self.env.config.md_link_adjust_affected_docs
        if self.env.docname not in affected_docs:
            # Skip processing if the document is not in the affected list
            return

        # Get the link replacement rules from the Sphinx configuration
        link_replacements = self.env.config.md_link_adjust_rule
        if not link_replacements:
            logger.warning("No link replacement rules specified in conf.py.")
            return

        logger.info(f"Adjusting links in document: {self.env.docname}")

        # Traverse through 'pending_xref' nodes (used by myst_parser for unresolved links)
        for node in list(self.document.traverse(pending_xref)):
            reftarget = node.get('reftarget')
            if reftarget:
                # Adjust link without adding an explicit file extension
                new_reftarget = self.replace_links(reftarget, link_replacements)
                if new_reftarget != reftarget:
                    logger.info(f"Replaced link: {reftarget} -> {new_reftarget}")
                    # Create a new reference node with updated link
                    reference_node = nodes.reference()
                    reference_node['refuri'] = new_reftarget
                    reference_node['classes'] = node.get('classes', [])

                    # Preserve the node’s children (text, formatting) in the new reference node
                    reference_node.extend(node.children)

                    # Replace the pending_xref node with the new reference node
                    node.replace_self(reference_node)

    def replace_links(self, uri, replacements):
        """
        Replace links in the URI based on the configured replacement rules,
        but do not add a specific file extension.
        """
        for old_link, new_link in replacements:
            if uri == old_link or re.match(old_link, uri):
                # Replace the link if there's an exact match or regex match
                updated_link = re.sub(old_link, new_link, uri)
                # Remove .rst extension without adding any other extension
                return re.sub(r'\.rst$', '', updated_link)
        return uri

def setup(app):
    app.add_transform(LinkAdjustTransform)
    # Add new config values for link replacement rules and affected documents
    app.add_config_value('md_link_adjust_rule', [], 'env')
    app.add_config_value('md_link_adjust_affected_docs', [], 'env')
