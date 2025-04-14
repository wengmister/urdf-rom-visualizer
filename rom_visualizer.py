import xml.etree.ElementTree as ET
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import argparse
import math
from itertools import product
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
    """Parse URDF file and extract joint information with limits."""
    tree_xml = ET.parse(file_path)
    root = tree_xml.getroot()
    
    # Dictionary to store links
    links = {}
    for link in root.findall('link'):
        links[link.attrib['name']] = link
    
    # Dictionary to store joints with parent-child relationships and limits
    joints = {}
    for joint in root.findall('joint'):
        joint_name = joint.attrib['name']
        joint_type = joint.attrib['type']
        parent = joint.find('parent').attrib['link']
        child = joint.find('child').attrib['link']
        origin_element = joint.find('origin')
        
        xyz, rpy = parse_origin(origin_element)
        
        # Default limits and axis
        lower_limit = 0.0
        upper_limit = 0.0
        axis = [1.0, 0.0, 0.0]  # Default axis is x-axis
        
        # Extract joint limits if available
        limit_element = joint.find('limit')
        if limit_element is not None:
            if 'lower' in limit_element.attrib:
                lower_limit = float(limit_element.attrib['lower'])
            if 'upper' in limit_element.attrib:
                upper_limit = float(limit_element.attrib['upper'])
        
        # Extract joint axis if available
        axis_element = joint.find('axis')
        if axis_element is not None and 'xyz' in axis_element.attrib:
            axis = [float(val) for val in axis_element.attrib['xyz'].split()]
            # Normalize the axis
            axis_norm = np.linalg.norm(axis)
            if axis_norm > 0:
                axis = [val / axis_norm for val in axis]
        
        # Movable (non‑fixed) joint flag
        movable = joint_type != 'fixed'
        
        joints[joint_name] = {
            'type': joint_type,
            'parent': parent,
            'child': child,
            'xyz': xyz,
            'rpy': rpy,
            'transform': transformation_matrix(xyz, rpy),
            'lower_limit': lower_limit,
            'upper_limit': upper_limit,
            'axis': axis,
            'movable': movable
        }
    
    return links, joints

def build_kinematic_tree(joints):
    """Build a dictionary representing the kinematic tree."""
    tree = {}
    root_link = None
    
    # Collect all links and determine which are children.
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
    
    # The root link is one that is not a child of any joint.
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

def rotation_matrix_from_axis_angle(axis, angle):
    """Create a rotation matrix from an axis and angle."""
    x, y, z = axis
    c = math.cos(angle)
    s = math.sin(angle)
    t = 1 - c
    
    return np.array([
        [t*x*x + c,    t*x*y - z*s,  t*x*z + y*s],
        [t*x*y + z*s,  t*y*y + c,    t*y*z - x*s],
        [t*x*z - y*s,  t*y*z + x*s,  t*z*z + c]
    ])

def calculate_joint_positions(joints, tree, root_link, joint_angles=None):
    """
    Calculate absolute positions for all joints.
    `joint_angles` should be a dictionary mapping joint name to its angle/displacement.
    If None, all joint angles default to 0.0.
    """
    joint_positions = {}
    link_transforms = {root_link: np.eye(4)}  # Start with the identity transform
    
    # Default joint angles to zero if not provided
    if joint_angles is None:
        joint_angles = {name: 0.0 for name in joints.keys()}
    
    def traverse_tree(link, parent_transform):
        if link in tree:
            for child, joint_name in tree[link]:
                joint_info = joints[joint_name]
                # Start with the fixed transform from the <origin>
                joint_transform = joint_info['transform'].copy()
                
                # If this joint is movable, apply its current angle/displacement.
                if joint_info['movable'] and joint_name in joint_angles:
                    angle = joint_angles[joint_name]
                    axis = joint_info['axis']
                    if joint_info['type'] in ['revolute', 'continuous']:
                        rotation = rotation_matrix_from_axis_angle(axis, angle)
                        # Update the rotation part of the transform
                        joint_transform[:3, :3] = np.dot(joint_transform[:3, :3], rotation)
                    elif joint_info['type'] == 'prismatic':
                        translation = np.array(axis) * angle
                        joint_transform[:3, 3] += translation
                
                # Compute absolute transform and store the joint's position.
                abs_transform = np.dot(parent_transform, joint_transform)
                joint_positions[joint_name] = abs_transform[:3, 3]
                link_transforms[child] = abs_transform
                
                # Recurse to process child links.
                traverse_tree(child, abs_transform)
    
    traverse_tree(root_link, link_transforms[root_link])
    return joint_positions, link_transforms

def sample_joint_positions(joints, tree, root_link, target_joints, joint_samples, num_samples):
    """Sample joint configurations and return positions of target joints."""
    movable_joints = [name for name in joint_samples.keys()]
    end_positions = []
    
    # Limit the number of combinations for performance
    max_combinations = 10000
    num_configs = 1
    for samples in joint_samples.values():
        num_configs *= len(samples)
    
    if num_configs > max_combinations:
        samples_per_joint = max(2, int(math.pow(max_combinations, 1/len(movable_joints))))
        print(f"Reducing samples from {num_samples} to {samples_per_joint} per joint due to combinatorial explosion")
        for joint_name in joint_samples:
            lower = joints[joint_name]['lower_limit']
            upper = joints[joint_name]['upper_limit']
            if joints[joint_name]['type'] == 'continuous':
                lower = -math.pi
                upper = math.pi
            joint_samples[joint_name] = np.linspace(lower, upper, samples_per_joint)
    
    count = 0
    print("Calculating workspace by sampling joint configurations...")
    for joint_values in product(*[joint_samples[j] for j in movable_joints]):
        joint_angles = dict(zip(movable_joints, joint_values))
        positions, _ = calculate_joint_positions(joints, tree, root_link, joint_angles)
        for joint_name in target_joints:
            if joint_name in positions:
                end_positions.append(positions[joint_name])
        count += 1
        if count % 100 == 0:
            print(f"Processed {count} configurations...")
        if count >= max_combinations:
            print(f"Reached maximum number of configurations ({max_combinations})")
            break
    
    print(f"Total configurations processed: {count}")
    return end_positions

def get_end_effector_positions(joints, tree, root_link, num_samples=50):
    """Sample joint configurations and return end effector positions."""
    movable_joints = [name for name, info in joints.items() if info['movable']]
    
    if not movable_joints:
        return []
    
    joint_samples = {}
    for joint_name in movable_joints:
        joint_info = joints[joint_name]
        lower = joint_info['lower_limit']
        upper = joint_info['upper_limit']
        if joint_info['type'] == 'continuous':
            lower = -math.pi
            upper = math.pi
        joint_samples[joint_name] = np.linspace(lower, upper, num_samples)
    
    # Identify end effectors as links that are not parents in the kinematic tree.
    link_with_children = set()
    for parent, children in tree.items():
        link_with_children.add(parent)
        for child, _ in children:
            link_with_children.add(child)
    
    all_links = set()
    for joint_info in joints.values():
        all_links.add(joint_info['parent'])
        all_links.add(joint_info['child'])
    
    end_effector_links = all_links - link_with_children
    
    if not end_effector_links:
        leaf_joints = []
        for joint_name, joint_info in joints.items():
            child = joint_info['child']
            if child not in tree:
                leaf_joints.append(joint_name)
        return sample_joint_positions(joints, tree, root_link, leaf_joints, joint_samples, num_samples)
    else:
        end_effector_joints = []
        for joint_name, joint_info in joints.items():
            if joint_info['child'] in end_effector_links:
                end_effector_joints.append(joint_name)
        return sample_joint_positions(joints, tree, root_link, end_effector_joints, joint_samples, num_samples)

def draw_scene(ax, joint_positions, tree, root_link, workspace_points=None):
    """Draw the robot (joints and connections) and workspace points on the given 3D axis."""
    ax.cla()
    # Plot joint positions
    xs = [pos[0] for pos in joint_positions.values()]
    ys = [pos[1] for pos in joint_positions.values()]
    zs = [pos[2] for pos in joint_positions.values()]
    ax.scatter(xs, ys, zs, c='b', marker='o', s=100, label='Joints')
    
    # Label joints
    for joint_name, position in joint_positions.items():
        ax.text(position[0], position[1], position[2], joint_name, size=8)
    
    # Draw lines connecting joints recursively
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
    
    # Plot workspace points if provided
    if workspace_points and len(workspace_points) > 0:
        ws_xs = [p[0] for p in workspace_points]
        ws_ys = [p[1] for p in workspace_points]
        ws_zs = [p[2] for p in workspace_points]
        ax.scatter(ws_xs, ws_ys, ws_zs, c='g', marker='.', s=10, alpha=0.2, label='Workspace')
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('URDF Joint Visualization with Reachable Workspace')
    ax.legend()
    
    all_points = []
    if joint_positions:
        all_points.extend(list(joint_positions.values()))
    if workspace_points:
        all_points.extend(workspace_points)
    if all_points:
        all_xs = [p[0] for p in all_points]
        all_ys = [p[1] for p in all_points]
        all_zs = [p[2] for p in all_points]
        max_range = max(max(all_xs)-min(all_xs), max(all_ys)-min(all_ys), max(all_zs)-min(all_zs))
        mid_x = (max(all_xs)+min(all_xs)) / 2
        mid_y = (max(all_ys)+min(all_ys)) / 2
        mid_z = (max(all_zs)+min(all_zs)) / 2
        ax.set_xlim(mid_x - max_range/2, mid_x + max_range/2)
        ax.set_ylim(mid_y - max_range/2, mid_y + max_range/2)
        ax.set_zlim(mid_z - max_range/2, mid_z + max_range/2)
    
    plt.draw()

def interactive_visualizer(links, joints, tree, root_link, workspace_points=None):
    """
    Launch an interactive visualizer:
      - The left 70% of the figure shows a 3D view of the robot and workspace.
      - The right 30% has vertical sliders to adjust each movable joint.
    """
    # Create one figure and reserve space on the right for sliders.
    fig = plt.figure(figsize=(12, 10))
    fig.subplots_adjust(right=0.7)
    ax = fig.add_subplot(111, projection='3d')
    
    # Initial joint angles: default to zero.
    initial_angles = {joint_name: 0.0 for joint_name, joint in joints.items() if joint['movable']}
    joint_positions, _ = calculate_joint_positions(joints, tree, root_link, initial_angles)
    draw_scene(ax, joint_positions, tree, root_link, workspace_points)
    
    # Create sliders for all movable joints.
    movable_joints = {name: joint for name, joint in joints.items() if joint['movable']}
    
    slider_height = 0.03
    slider_padding = 0.01
    sliders = {}
    
    for i, (joint_name, joint) in enumerate(movable_joints.items()):
        left = 0.72    # Right-side region for sliders.
        width = 0.22
        bottom = 0.95 - (i+1) * (slider_height + slider_padding)
        rect = [left, bottom, width, slider_height]
        ax_slider = fig.add_axes(rect)
        slider = Slider(ax_slider,
                        joint_name,
                        joint['lower_limit'],
                        joint['upper_limit'],
                        valinit=0.0)
        sliders[joint_name] = slider

    def update_sliders(val):
        current_angles = {name: slider.val for name, slider in sliders.items()}
        joint_positions, _ = calculate_joint_positions(joints, tree, root_link, current_angles)
        draw_scene(ax, joint_positions, tree, root_link, workspace_points)
    
    for slider in sliders.values():
        slider.on_changed(update_sliders)
    
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='Visualize joints, workspace, and kinematic tree from a URDF file.')
    parser.add_argument('urdf_file', type=str, help='Path to the URDF file')
    parser.add_argument('--samples', type=int, default=200, help='Number of samples per joint for workspace analysis')
    parser.add_argument('--no-workspace', action='store_true', help='Skip workspace analysis')
    args = parser.parse_args()
    
    print(f"Parsing URDF file: {args.urdf_file}")
    links, joints = parse_urdf(args.urdf_file)
    
    print("Building kinematic tree...")
    tree, root_link = build_kinematic_tree(joints)
    print(f"Root link: {root_link}")
    

    
    print("Calculating joint positions for neutral pose...")
    joint_positions, _ = calculate_joint_positions(joints, tree, root_link)
    
    workspace_points = []
    if not args.no_workspace:
        print(f"Calculating workspace using {args.samples} samples per joint...")
        workspace_points = get_end_effector_positions(joints, tree, root_link, args.samples)
        print(f"Workspace analysis complete: {len(workspace_points)} points generated")

    print("\nKinematic Tree:")
    print_kinematic_tree(tree, root_link)
    
    print("Launching interactive visualizer (with joint sliders)...")
    interactive_visualizer(links, joints, tree, root_link, workspace_points)



if __name__ == "__main__":
    main()
