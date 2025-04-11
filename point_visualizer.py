import xml.etree.ElementTree as ET
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import argparse
import math

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
    
    # Roll (rotation around X-axis)
    Rx = np.array([
        [1, 0, 0],
        [0, math.cos(roll), -math.sin(roll)],
        [0, math.sin(roll), math.cos(roll)]
    ])
    
    # Pitch (rotation around Y-axis)
    Ry = np.array([
        [math.cos(pitch), 0, math.sin(pitch)],
        [0, 1, 0],
        [-math.sin(pitch), 0, math.cos(pitch)]
    ])
    
    # Yaw (rotation around Z-axis)
    Rz = np.array([
        [math.cos(yaw), -math.sin(yaw), 0],
        [math.sin(yaw), math.cos(yaw), 0],
        [0, 0, 1]
    ])
    
    # Combined rotation matrix (R = Rz * Ry * Rx)
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
    
    # Dictionary to store links
    links = {}
    for link in root.findall('link'):
        links[link.attrib['name']] = link
    
    # Dictionary to store joints with parent-child relationships
    joints = {}
    for joint in root.findall('joint'):
        joint_name = joint.attrib['name']
        joint_type = joint.attrib['type']
        parent = joint.find('parent').attrib['link']
        child = joint.find('child').attrib['link']
        origin_element = joint.find('origin')
        
        xyz, rpy = parse_origin(origin_element)
        
        joints[joint_name] = {
            'type': joint_type,
            'parent': parent,
            'child': child,
            'xyz': xyz,
            'rpy': rpy,
            'transform': transformation_matrix(xyz, rpy)
        }
    
    return links, joints

def build_kinematic_tree(joints):
    """Build a dictionary representing the kinematic tree."""
    tree = {}
    root_link = None
    
    # Find all links
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
    
    # Find the root link (the one that's not a child of any joint)
    root_candidates = all_links - child_links
    if root_candidates:
        root_link = next(iter(root_candidates))
    
    return tree, root_link

def calculate_joint_positions(joints, tree, root_link):
    """Calculate absolute positions of all joints."""
    joint_positions = {}
    link_transforms = {root_link: np.eye(4)}  # Start with identity transform for root
    
    def traverse_tree(link, parent_transform):
        if link in tree:
            for child, joint_name in tree[link]:
                joint_transform = joints[joint_name]['transform']
                # Calculate absolute transform
                abs_transform = np.dot(parent_transform, joint_transform)
                # Store the joint position (translation part of the transform)
                joint_positions[joint_name] = abs_transform[:3, 3]
                # Update the child link transform
                link_transforms[child] = abs_transform
                # Recursively process children
                traverse_tree(child, abs_transform)
    
    traverse_tree(root_link, link_transforms[root_link])
    return joint_positions

def visualize_joints(joint_positions, joints, tree, root_link):
    """Visualize the joints and their connections."""
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot joints as points
    xs = [pos[0] for pos in joint_positions.values()]
    ys = [pos[1] for pos in joint_positions.values()]
    zs = [pos[2] for pos in joint_positions.values()]
    
    # Plot points for each joint
    ax.scatter(xs, ys, zs, c='b', marker='o', s=100)
    
    # Add joint names as text
    for joint_name, position in joint_positions.items():
        ax.text(position[0], position[1], position[2], joint_name, size=8)
    
    # Draw lines between connected joints
    def draw_connections(link, parent_joint=None):
        if link in tree:
            for child, joint_name in tree[link]:
                if parent_joint and joint_name in joint_positions:
                    # Draw a line from parent joint to this joint
                    start_pos = joint_positions[parent_joint]
                    end_pos = joint_positions[joint_name]
                    ax.plot([start_pos[0], end_pos[0]],
                            [start_pos[1], end_pos[1]],
                            [start_pos[2], end_pos[2]], 'r-')
                
                # Continue recursion
                draw_connections(child, joint_name)
    
    # Draw connections starting from children of root
    if root_link in tree:
        for child, joint_name in tree[root_link]:
            if joint_name in joint_positions:
                # For root's children, start line from origin
                end_pos = joint_positions[joint_name]
                ax.plot([0, end_pos[0]],
                        [0, end_pos[1]],
                        [0, end_pos[2]], 'r-')
            draw_connections(child, joint_name)
    
    # Set labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('URDF Joint Visualization')
    
    # Make the plot look better
    max_range = max(
        max(xs) - min(xs),
        max(ys) - min(ys),
        max(zs) - min(zs)
    )
    mid_x = (max(xs) + min(xs)) / 2
    mid_y = (max(ys) + min(ys)) / 2
    mid_z = (max(zs) + min(zs)) / 2
    ax.set_xlim(mid_x - max_range/2, mid_x + max_range/2)
    ax.set_ylim(mid_y - max_range/2, mid_y + max_range/2)
    ax.set_zlim(mid_z - max_range/2, mid_z + max_range/2)
    
    plt.tight_layout()
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='Visualize joints from a URDF file.')
    parser.add_argument('urdf_file', type=str, help='Path to the URDF file')
    args = parser.parse_args()
    
    print(f"Parsing URDF file: {args.urdf_file}")
    links, joints = parse_urdf(args.urdf_file)
    
    print("Building kinematic tree...")
    tree, root_link = build_kinematic_tree(joints)
    print(f"Root link: {root_link}")
    
    print("Calculating joint positions...")
    joint_positions = calculate_joint_positions(joints, tree, root_link)
    
    print("Visualizing joints...")
    visualize_joints(joint_positions, joints, tree, root_link)

if __name__ == "__main__":
    main()