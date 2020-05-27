from xml.etree import ElementTree
from graphviz import Digraph

LINK_COLOR = "blue"
JOINT_COLOR = "green"


def _get_next_joints(root, current_link):
    current_link_name = current_link.attrib["name"]
    next_joints = []
    for joint in root.findall("joint"):
        if joint.find("parent").attrib["link"] == current_link_name:
            next_joints.append(joint)

    return next_joints


def _get_next_links(root, current_joint):
    child_link_name = current_joint.findall("child")[0].attrib['link']
    child_link = None
    for link in root.findall("link"):
        if link.attrib["name"] == child_link_name:
            child_link = link
            break

    return [child_link]


class URDFTree:
    """
    Utility class to represent a URDF tree, only used here
    Still very experimental, this class will change in the future
    """
    def __init__(self, name):
        self.name = name
        self.children_links = {}

    def __repr__(self):
        return "URDF Link: {};\n".format(self.name)


def _create_robot_tree_aux(dot, root, current_link, current_robot_link):
    """

    Parameters
    ----------
    dot: Digraph
    root
    current_link
    current_robot_link

    Returns
    -------

    """
    print(root)
    # TODO: This implementation could be greatly improved
    # Instead of doing a research of child links for each of the links and passing through all of the links (O(n^2))
    # We should instead parse each links and create their child/parent relationship once
    # For example in a dict
    # And with this we just a have to go trough this dict to build the _URDFLInk data structure
    for next_joint in _get_next_joints(root, current_link):
        print(current_link)

        # NOTE: We need to create IDs that are unique to each joint/link and use them as node ids
        # Actually, some joints and links can have the same name, and they would appear as the same node in the final graph

        # Get next joint

        next_joint_id = "joint_" + next_joint.attrib["name"]
        dot.node(next_joint_id, label=next_joint.attrib["name"], color=JOINT_COLOR, fillcolor="lightgrey", style="filled")
        dot.edge("link_" + current_link.attrib["name"], next_joint_id)

        # Get link associated with each joint
        next_link = _get_next_links(root, next_joint)[0]
        next_robot_link = URDFTree(name=next_link.attrib["name"])
        current_robot_link.children_links[next_link.attrib["name"]] = next_robot_link
        next_link_id = "link_" + next_link.attrib["name"]
        dot.node(next_link_id, label=next_link.attrib["name"], shape="box", color=LINK_COLOR, fillcolor="lightgrey", style="filled")
        dot.edge(next_joint_id, next_link_id)
        # print("{} is son of {} ".format(next_link.attrib["name"], current_link.attrib["name"]))

        if next_link is not None:
            _create_robot_tree_aux(dot, root, next_link, next_robot_link)


def get_urdf_tree(urdf_path, out_image_path=None, root_element="base", legend=False):
    """
    Parse an URDF file into a tree of links

    Parameters
    ----------
    urdf_path: str
        Path towards the URDF file
    out_image_path: str
        If set, save the graph as a pdf in `out_image_path`
    root_element: str
        name of the element that will be used as the root of the tree. Common to be "base"

    legend: bool
        Add a legend to the final graph

    Returns
    -------
    dot: graphviz.Digraph
        The rendered plot
    urdf_tree: URDFTree

    """
    tree = ElementTree.parse(urdf_path)
    root = tree.getroot()

    base_link = root.find("link[@name='{}']".format(root_element))
    urdf_tree = URDFTree(root_element)

    # Initialize the rec
    dot = Digraph(name="robot")
    dot.node("link_" + root_element, label=root_element, shape="box", color=LINK_COLOR, fillcolor="lightgrey", style="filled")

    # Parse the tree
    _create_robot_tree_aux(dot, root, base_link, urdf_tree)

    if legend:
        # Add a little legend
        with dot.subgraph(name="cluster_legend") as legend:
            legend.attr(style="filled", fillcolor="grey", rankdir="TB")
            legend.attr(label='legend')
            legend.node("Link", style="filled", color=LINK_COLOR, shape="square", fillcolor="lightgrey", rank="same")
            legend.node("Joint", style="filled", color=JOINT_COLOR, fillcolor="lightgrey", rank="same")

    # Finally render the tree
    if out_image_path is not None:
        dot.render(out_image_path)

    return dot, urdf_tree
