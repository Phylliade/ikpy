from ikpy.urdf.utils import get_urdf_tree

# Generate URDF Tree PDF
dot, urdf_tree = get_urdf_tree('robotis_op3.urdf', out_image_path='robotis_op3', root_element='body_link')