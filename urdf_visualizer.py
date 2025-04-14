import xml.etree.ElementTree as ET
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import argparse
import math
from matplotlib.widgets import Slider

def parse_origin(origin_element):
    """Parse the origin element to get position and rotation."""
    xyz = [0.0, 0.0, 0.0]
    rpy = [0.0, 0.0, 0.0]
    if origin_element is not None:
        if 'xyz' in origin_element.attrib:
            xyz = [float(val) for val in origin_element.attrib['xyz'].split()]
        if 'rpy' in origin_element.attrib:
            rpy = [float(val) for val in origin_element.attrib['rpy'].split()]
    return xyz, rpy

def rotation_matrix(rpy):
    """Create a rotation matrix from roll, pitch, yaw angles."""
    roll, pitch, yaw = rpy
    Rx = np.array([
        [1, 0, 0],
        [0, math.cos(roll), -math.sin(roll)],
        [0, math.sin(roll), math.cos(roll)]
    ])
    Ry = np.array([
        [math.cos(pitch), 0, math.sin(pitch)],
        [0, 1, 0],
        [-math.sin(pitch), 0, math.cos(pitch)]
    ])
    Rz = np.array([
        [math.cos(yaw), -math.sin(yaw), 0],
        [math.sin(yaw), math.cos(yaw), 0],
        [0, 0, 1]
    ])
    R = np.dot(Rz, np.dot(Ry, Rx))
    return R

def transformation_matrix(xyz, rpy):
    """Create a 4x4 transformation matrix from position and rotation."""
    R = rotation_matrix(rpy)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = xyz
    return T

def parse_urdf(file_path):
    """Parse URDF file and extract joint information."""
    tree = ET.parse(file_path)
    root = tree.getroot()

    # Store links by name (if needed for further processing)
    links = {}
    for link in root.findall('link'):
        links[link.attrib['name']] = link

    # Process joints
    joints = {}
    for joint in root.findall('joint'):
        joint_name = joint.attrib['name']
        joint_type = joint.attrib['type']
        parent = joint.find('parent').attrib['link']
        child = joint.find('child').attrib['link']
        origin_element = joint.find('origin')
        xyz, rpy = parse_origin(origin_element)
        
        # Compute the static (fixed) transformation from the origin.
        joint_info = {
            'type': joint_type,
            'parent': parent,
            'child': child,
            'xyz': xyz,
            'rpy': rpy,
            'transform': transformation_matrix(xyz, rpy)
        }
        
        # For movable joints, parse the axis and limits
        if joint_type in ['revolute', 'prismatic']:
            axis_element = joint.find('axis')
            if axis_element is not None and 'xyz' in axis_element.attrib:
                axis = [float(val) for val in axis_element.attrib['xyz'].split()]
            else:
                axis = [0, 0, 1]  # Default axis (typically the z-axis in URDF)
            joint_info['axis'] = np.array(axis, dtype=float)
            joint_info['q'] = 0.0  # initial joint variable (angle or displacement)
            
            # Parse limits if provided; otherwise default to [-pi, pi]
            limit_element = joint.find('limit')
            if limit_element is not None:
                lower = float(limit_element.attrib.get('lower', -math.pi))
                upper = float(limit_element.attrib.get('upper', math.pi))
            else:
                lower = -math.pi
                upper = math.pi
            joint_info['limit'] = {'lower': lower, 'upper': upper}
        
        joints[joint_name] = joint_info
    return links, joints

def build_kinematic_tree(joints):
    """Build a dictionary representing the kinematic tree."""
    tree = {}
    root_link = None
    all_links = set()
    child_links = set()
    
    for joint_name, joint_info in joints.items():
        parent = joint_info['parent']
        child = joint_info['child']
        all_links.add(parent)
        all_links.add(child)
        child_links.add(child)
        if parent not in tree:
            tree[parent] = []
        tree[parent].append((child, joint_name))
    
    # The root link is one that is never a child.
    root_candidates = all_links - child_links
    if root_candidates:
        root_link = next(iter(root_candidates))
    
    return tree, root_link

def print_kinematic_tree(tree, current_link, indent=0):
    """
    Recursively print the kinematic tree.
    Each level is indented for clarity.
    """
    indent_str = "  " * indent
    print(f"{indent_str}{current_link}")
    if current_link in tree:
        for child, joint_name in tree[current_link]:
            print(f"{indent_str}  |--({joint_name})--> {child}")
            print_kinematic_tree(tree, child, indent + 2)

def rotation_about_axis(axis, angle):
    """Compute a 4x4 rotation matrix about an arbitrary axis using Rodrigues' formula."""
    axis = np.array(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    c = np.cos(angle)
    s = np.sin(angle)
    t = 1 - c
    x, y, z = axis
    R = np.array([
        [t*x*x + c,   t*x*y - s*z, t*x*z + s*y],
        [t*x*y + s*z, t*y*y + c,   t*y*z - s*x],
        [t*x*z - s*y, t*y*z + s*x, t*z*z + c]
    ])
    T = np.eye(4)
    T[:3, :3] = R
    return T

def translation_along_axis(axis, displacement):
    """Compute a 4x4 translation matrix along a given axis."""
    axis = np.array(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    T = np.eye(4)
    T[:3, 3] = axis * displacement
    return T

def compute_joint_transform(joint):
    """
    Compute the full transformation for a joint.
    This is the product of the fixed transform (from the <origin>)
    and the motion transform computed from the joint state.
    """
    T_origin = joint['transform']
    if joint['type'] == 'revolute':
        T_motion = rotation_about_axis(joint['axis'], joint['q'])
    elif joint['type'] == 'prismatic':
        T_motion = translation_along_axis(joint['axis'], joint['q'])
    else:  # For fixed joints, no additional motion
        T_motion = np.eye(4)
    return np.dot(T_origin, T_motion)

def calculate_joint_positions(joints, tree, root_link):
    """Calculate absolute positions of all joints based on the current joint states."""
    joint_positions = {}
    link_transforms = {root_link: np.eye(4)}
    
    def traverse_tree(link, parent_transform):
        if link in tree:
            for child, joint_name in tree[link]:
                joint = joints[joint_name]
                joint_transform = compute_joint_transform(joint)
                abs_transform = np.dot(parent_transform, joint_transform)
                joint_positions[joint_name] = abs_transform[:3, 3]
                link_transforms[child] = abs_transform
                traverse_tree(child, abs_transform)
                
    traverse_tree(root_link, link_transforms[root_link])
    return joint_positions

def draw_robot(ax, joint_positions, tree, root_link):
    """Draw the robot's joints and connections on the given 3D axis."""
    ax.cla()  # Clear the axis for redrawing
    # Plot joint positions as blue dots.
    xs = [pos[0] for pos in joint_positions.values()]
    ys = [pos[1] for pos in joint_positions.values()]
    zs = [pos[2] for pos in joint_positions.values()]
    ax.scatter(xs, ys, zs, c='b', marker='o', s=100)
    
    # Label each joint.
    for joint_name, position in joint_positions.items():
        ax.text(position[0], position[1], position[2], joint_name, size=8)
    
    # Draw lines connecting joints.
    def draw_connections(link, parent_joint=None):
        if link in tree:
            for child, joint_name in tree[link]:
                if parent_joint is not None and joint_name in joint_positions:
                    start_pos = joint_positions[parent_joint]
                    end_pos = joint_positions[joint_name]
                    ax.plot([start_pos[0], end_pos[0]],
                            [start_pos[1], end_pos[1]],
                            [start_pos[2], end_pos[2]], 'r-')
                draw_connections(child, joint_name)
    
    if root_link in tree:
        for child, joint_name in tree[root_link]:
            if joint_name in joint_positions:
                end_pos = joint_positions[joint_name]
                ax.plot([0, end_pos[0]],
                        [0, end_pos[1]],
                        [0, end_pos[2]], 'r-')
            draw_connections(child, joint_name)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('URDF Joint Visualization')
    
    # Adjust the axis limits for a nicer view.
    if xs and ys and zs:
        max_range = max(max(xs)-min(xs), max(ys)-min(ys), max(zs)-min(zs))
        mid_x = (max(xs)+min(xs)) / 2
        mid_y = (max(ys)+min(ys)) / 2
        mid_z = (max(zs)+min(zs)) / 2
        ax.set_xlim(mid_x - max_range/2, mid_x + max_range/2)
        ax.set_ylim(mid_y - max_range/2, mid_y + max_range/2)
        ax.set_zlim(mid_z - max_range/2, mid_z + max_range/2)
    plt.draw()

def interactive_visualizer(links, joints, tree, root_link):
    """
    Launch an interactive visualizer:
      - A 3D view of the robot is shown on the left.
      - For each movable joint, a slider is added on the right side.
    """
    # Create a single figure and adjust its layout.
    fig = plt.figure(figsize=(12, 10))
    fig.subplots_adjust(right=0.7)
    ax = fig.add_subplot(111, projection='3d')
    
    # Draw the initial robot configuration.
    joint_positions = calculate_joint_positions(joints, tree, root_link)
    draw_robot(ax, joint_positions, tree, root_link)
    
    # Select movable joints.
    movable_joints = {name: joint for name, joint in joints.items() if joint['type'] in ['revolute', 'prismatic']}
    
    # Create vertical sliders on the right side.
    slider_height = 0.03
    slider_padding = 0.01
    sliders = {}
    for i, (joint_name, joint) in enumerate(movable_joints.items()):
        left = 0.72
        width = 0.22
        bottom = 0.95 - (i+1)*(slider_height + slider_padding)
        rect = [left, bottom, width, slider_height]
        ax_slider = fig.add_axes(rect)
        slider = Slider(ax_slider,
                        joint_name,
                        joint['limit']['lower'],
                        joint['limit']['upper'],
                        valinit=0.0)
        sliders[joint_name] = slider

    def update_sliders(val):
        """Update joint states and re-render the robot."""
        for joint_name, slider in sliders.items():
            joints[joint_name]['q'] = slider.val
        new_positions = calculate_joint_positions(joints, tree, root_link)
        draw_robot(ax, new_positions, tree, root_link)
    
    # Connect slider events.
    for slider in sliders.values():
        slider.on_changed(update_sliders)
    
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='Interactive URDF Joint Visualizer')
    parser.add_argument('urdf_file', type=str, help='Path to the URDF file')
    args = parser.parse_args()
    
    print(f"Parsing URDF file: {args.urdf_file}")
    links, joints = parse_urdf(args.urdf_file)
    
    print("Building kinematic tree...")
    tree, root_link = build_kinematic_tree(joints)
    print(f"Root link: {root_link}")
    
    print("\nKinematic Tree:")
    print_kinematic_tree(tree, root_link)
    
    print("Launching interactive visualizer...")
    interactive_visualizer(links, joints, tree, root_link)

if __name__ == "__main__":
    main()
