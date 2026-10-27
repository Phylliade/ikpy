# coding= utf8
"""
Selection of a kinematic chain inside a tree: the common vocabulary of the URDF, MJCF and BVH parsers.

A robot description is a tree. A chain is the path from a ``base`` element to a ``tip`` element, and in a tree
that path is unique, so the two names are all it takes to select it. ``tip_offset`` adds a fixed translation at
the end of the chain, in the frame of its last joint.

The former way of selecting a chain, ``base_elements`` (an ordered list of names, completed by following the
first child of each element) with ``base_element_type`` and ``last_link_vector``, is deprecated and will be
removed in IKPy 5.0. The functions here translate it and keep its behaviour meanwhile.
"""
import warnings


def resolve_selection(base=None, tip=None, tip_offset=None, base_elements=None, last_link_vector=None, base_element_type=None, stacklevel=3):
    """Translate the deprecated selection arguments, and check that the two forms are not mixed

    Returns
    -------
    base, tip, tip_offset, legacy
        ``legacy`` is None with the new form, or the tuple ``(base_elements, base_element_type)`` with the old one
    """
    new_form = base is not None or tip is not None or tip_offset is not None
    old_form = base_elements is not None or last_link_vector is not None or base_element_type is not None
    if new_form and old_form:
        raise ValueError("Give the chain with base, tip and tip_offset, or with the deprecated base_elements, base_element_type and last_link_vector, not both")

    if last_link_vector is not None:
        warnings.warn("last_link_vector is deprecated and will be removed in IKPy 5.0: use tip_offset, which means the same thing", DeprecationWarning, stacklevel=stacklevel)
        tip_offset = last_link_vector

    legacy = None
    if base_elements is not None or base_element_type is not None:
        hint = "base='{}'".format(base_elements[0]) if base_elements else "base=..."
        warnings.warn(
            "base_elements and base_element_type are deprecated and will be removed in IKPy 5.0: select the chain with "
            "{} and tip=..., the path between the two is found automatically (see Chain.from_urdf_file)".format(hint),
            DeprecationWarning, stacklevel=stacklevel)
        legacy = (base_elements, base_element_type if base_element_type is not None else "link")

    return base, tip, tip_offset, legacy


def path_between(parent_of, base, tip, display=str):
    """Return the nodes from base to tip, both included, in a tree given as a child -> parent mapping

    Raises a ValueError if tip is not below base. `display` turns a node into the name shown in that error.
    """
    path = [tip]
    while path[-1] != base:
        parent = parent_of.get(path[-1])
        if parent is None:
            raise ValueError("'{}' is not below '{}' in the tree: the chain must go from an element to one of its descendants".format(display(tip), display(base)))
        path.append(parent)
    return path[::-1]


def leaves_below(children_of, base):
    """Return the names of the leaves below a node, in depth-first order, the node itself if it is a leaf"""
    leaves = []
    stack = [base]
    while stack:
        node = stack.pop()
        children = children_of.get(node, [])
        if not children:
            leaves.append(node)
        else:
            stack.extend(reversed(children))
    return leaves


def unique_leaf(children_of, base, kind, display=str):
    """Return the only leaf below a node, or raise a ValueError listing the leaves to choose a tip from"""
    leaves = leaves_below(children_of, base)
    if len(leaves) != 1:
        raise ValueError("Several {} end the tree below '{}': {}. Choose the end of the chain with tip=...".format(kind, display(base), [display(leaf) for leaf in leaves]))
    return leaves[0]


def unique_root(roots, kind, display=str):
    """Return the only root of a tree, or raise a ValueError listing them to choose a base from"""
    if len(roots) != 1:
        raise ValueError("The file has {} root {}: {}. Choose the start of the chain with base=...".format(len(roots), kind, [display(root) for root in roots]))
    return roots[0]
