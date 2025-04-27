import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import xml.etree.ElementTree as ET
import random
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation as R
import math

# Add necessary functions for manipulability and joint limit distance calculations
def calculate_manipulability(joint_limits, joint_values, method='yoshikawa'):
    """
    Calculate the manipulability measure at the given joint configuration.
    This is a simplified implementation based on joint proximity to limits.
    
    A more accurate implementation would compute the Jacobian matrix and 
    use the Yoshikawa manipulability measure: sqrt(det(J*J^T))
    """
    # Simplified implementation - uses joint positions relative to their limits
    manipulability = 1.0
    
    for joint_name, joint_value in joint_values.items():
        if joint_name in joint_limits and 'type' not in joint_limits[joint_name]:
            lower = joint_limits[joint_name]['lower']
            upper = joint_limits[joint_name]['upper']
            
            # Calculate normalized position in joint range (0 to 1)
            if upper > lower:
                normalized_pos = (joint_value - lower) / (upper - lower)
                
                # Calculate distance from center of joint range
                # (0.5 is the center, 0 means at a limit, 1 means at the other limit)
                center_dist = abs(normalized_pos - 0.5) * 2  # Scale to [0,1]
                
                # This factor will be closer to 1 when joint is near middle of range
                # and closer to 0 when joint is near limits
                factor = 1.0 - center_dist
                
                # Accumulate the manipulability
                manipulability *= (0.5 + factor)  # Scale to make sure it's never exactly 0
    
    return manipulability

def calculate_distance_from_limits(joint_limits, joint_values):
    """
    Calculate a measure of how far the joint configuration is from its limits.
    Returns a value between 0 and 1, where:
    - 1 means all joints are at the center of their ranges
    - 0 means at least one joint is at its limit
    """
    min_distance = 1.0
    
    for joint_name, joint_value in joint_values.items():
        if joint_name in joint_limits and 'type' not in joint_limits[joint_name]:
            lower = joint_limits[joint_name]['lower']
            upper = joint_limits[joint_name]['upper']
            
            # Skip if joint limits are the same (shouldn't happen)
            if upper == lower:
                continue
                
            # Calculate normalized position in joint range (0 to 1)
            normalized_pos = (joint_value - lower) / (upper - lower)
            
            # Calculate distance from nearest limit (0 means at a limit)
            distance = min(normalized_pos, 1 - normalized_pos) * 2  # Scale to [0,1]
            
            # Keep track of the minimum distance across all joints
            min_distance = min(min_distance, distance)
    
    return min_distance

# Parse URDF file
def parse_urdf(file_path):
    tree = ET.parse(file_path)
    root = tree.getroot()
    
    # Dictionary to store joint limits
    joint_limits = {}
    
    # Dictionary to store link transformations
    link_transforms = {}
    
    # Parse all joints
    for joint in root.findall('.//joint'):
        joint_name = joint.get('name')
        
        if joint.get('type') == 'revolute':
            # Get parent and child links
            parent = joint.find('parent').get('link')
            child = joint.find('child').get('link')
            
            # Get joint axis
            axis_element = joint.find('axis')
            if axis_element is not None:
                axis = np.array([float(x) for x in axis_element.get('xyz').split()])
            else:
                axis = np.array([0, 0, 1])  # Default axis
            
            # Get origin transformation
            origin = joint.find('origin')
            xyz = np.array([0, 0, 0])
            rpy = np.array([0, 0, 0])
            
            if origin is not None:
                if origin.get('xyz'):
                    xyz = np.array([float(x) for x in origin.get('xyz').split()])
                if origin.get('rpy'):
                    rpy = np.array([float(x) for x in origin.get('rpy').split()])
            
            # Get joint limits
            limit = joint.find('limit')
            if limit is not None:
                lower = float(limit.get('lower'))
                upper = float(limit.get('upper'))
            else:
                lower = -np.pi
                upper = np.pi
                
            # Store joint information
            joint_limits[joint_name] = {
                'parent': parent,
                'child': child,
                'axis': axis,
                'origin_xyz': xyz,
                'origin_rpy': rpy,
                'lower': lower,
                'upper': upper
            }
            
        elif joint.get('type') == 'fixed':
            # Get parent and child links
            parent = joint.find('parent').get('link')
            child = joint.find('child').get('link')
            
            # Get origin transformation
            origin = joint.find('origin')
            xyz = np.array([0, 0, 0])
            rpy = np.array([0, 0, 0])
            
            if origin is not None:
                if origin.get('xyz'):
                    xyz = np.array([float(x) for x in origin.get('xyz').split()])
                if origin.get('rpy'):
                    rpy = np.array([float(x) for x in origin.get('rpy').split()])
            
            # Store joint information without limits for fixed joints
            joint_limits[joint_name] = {
                'parent': parent,
                'child': child,
                'origin_xyz': xyz,
                'origin_rpy': rpy,
                'type': 'fixed'
            }
    
    return joint_limits

# Create transformation matrix from translation and rotation
def transformation_matrix(trans, rot_rpy):
    # Create rotation matrix from roll, pitch, yaw
    rx, ry, rz = rot_rpy
    
    # Create rotation matrices for roll, pitch, yaw
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(rx), -np.sin(rx)],
        [0, np.sin(rx), np.cos(rx)]
    ])
    
    Ry = np.array([
        [np.cos(ry), 0, np.sin(ry)],
        [0, 1, 0],
        [-np.sin(ry), 0, np.cos(ry)]
    ])
    
    Rz = np.array([
        [np.cos(rz), -np.sin(rz), 0],
        [np.sin(rz), np.cos(rz), 0],
        [0, 0, 1]
    ])
    
    # Combine rotation matrices
    R_combined = Rz @ Ry @ Rx
    
    # Create 4x4 transformation matrix
    T = np.eye(4)
    T[:3, :3] = R_combined
    T[:3, 3] = trans
    
    return T

# Create rotation matrix around arbitrary axis
def rotation_matrix_from_axis_angle(axis, angle):
    # Normalize axis
    axis = axis / np.linalg.norm(axis)
    
    # Create rotation matrix using Rodrigues' formula
    c = np.cos(angle)
    s = np.sin(angle)
    t = 1 - c
    x, y, z = axis
    
    R = np.array([
        [t*x*x + c, t*x*y - z*s, t*x*z + y*s],
        [t*x*y + z*s, t*y*y + c, t*y*z - x*s],
        [t*x*z - y*s, t*y*z + x*s, t*z*z + c]
    ])
    
    return R

# Calculate forward kinematics for a given joint configuration
def forward_kinematics(joint_limits, joint_values, end_effector_link):
    # Create transformation matrices for each joint
    transforms = {}
    transforms['base'] = np.eye(4)
    
    # Topologically sort links
    sorted_links = ['base']
    link_to_joint = {}
    
    # Create mapping from child link to joint
    for joint_name, joint_info in joint_limits.items():
        link_to_joint[joint_info['child']] = joint_name
    
    # Topological sort
    while len(sorted_links) < len(link_to_joint) + 1:
        for joint_name, joint_info in joint_limits.items():
            if joint_info['parent'] in sorted_links and joint_info['child'] not in sorted_links:
                sorted_links.append(joint_info['child'])
    
    # Calculate transformations
    for link in sorted_links[1:]:  # Skip base
        joint_name = link_to_joint[link]
        joint_info = joint_limits[joint_name]
        
        # Get parent transformation
        parent_T = transforms[joint_info['parent']]
        
        # Create joint transformation
        joint_origin_T = transformation_matrix(joint_info['origin_xyz'], joint_info['origin_rpy'])
        
        if 'type' in joint_info and joint_info['type'] == 'fixed':
            # For fixed joints, no additional rotation
            joint_T = joint_origin_T
        else:
            # For revolute joints, apply rotation around axis
            angle = joint_values[joint_name]
            axis = joint_info['axis']
            
            # Create rotation matrix
            R = rotation_matrix_from_axis_angle(axis, angle)
            
            # Create transformation matrix
            joint_rot_T = np.eye(4)
            joint_rot_T[:3, :3] = R
            
            # Combine origin transformation and rotation
            joint_T = joint_origin_T @ joint_rot_T
        
        # Combine with parent transformation
        transforms[link] = parent_T @ joint_T
    
    # Return end effector transformation
    return transforms[end_effector_link]

# Sample joint configurations within limits using a combined random and gradient-based approach
def sample_joint_configurations(joint_limits, num_samples, use_gradient=True):
    # Initialize joint values dictionary
    joint_values_list = []
    
    # First, generate some random samples to explore the space
    num_random_samples = num_samples // 2
    
    for _ in range(num_random_samples):
        joint_values = {}
        
        for joint_name, joint_info in joint_limits.items():
            if 'type' not in joint_info:  # Only for revolute joints
                lower = joint_info['lower']
                upper = joint_info['upper']
                
                # Sample random value within limits
                value = random.uniform(lower, upper)
                joint_values[joint_name] = value
        
        joint_values_list.append(joint_values)
    
    if use_gradient and num_random_samples > 0:
        # Use the valid random samples as seeds for gradient-based optimization
        valid_configs = []
        for config in joint_values_list:
            # Test if this configuration is valid (would require pre-implementing some evaluation)
            # For now, we'll just add all initial samples to the valid list
            valid_configs.append(config)
        
        # If we have valid seeds, generate variations around them
        if valid_configs:
            variations_per_seed = (num_samples - num_random_samples) // len(valid_configs)
            
            for seed_config in valid_configs:
                for _ in range(variations_per_seed):
                    # Create a variation of the seed configuration
                    new_config = {}
                    
                    for joint_name, joint_value in seed_config.items():
                        if joint_name in joint_limits and 'type' not in joint_limits[joint_name]:
                            lower = joint_limits[joint_name]['lower']
                            upper = joint_limits[joint_name]['upper']
                            
                            # Add a small random perturbation (gradually smaller as we iterate)
                            perturbation = random.uniform(-0.2, 0.2)  # Smaller perturbation
                            new_value = joint_value + perturbation
                            
                            # Ensure we stay within joint limits
                            new_value = max(lower, min(upper, new_value))
                            new_config[joint_name] = new_value
                    
                    joint_values_list.append(new_config)
    
    # If we have fewer samples than requested (due to no valid seeds), add more random samples
    while len(joint_values_list) < num_samples:
        joint_values = {}
        
        for joint_name, joint_info in joint_limits.items():
            if 'type' not in joint_info:  # Only for revolute joints
                lower = joint_info['lower']
                upper = joint_info['upper']
                
                # Sample random value within limits
                value = random.uniform(lower, upper)
                joint_values[joint_name] = value
        
        joint_values_list.append(joint_values)
    
    return joint_values_list

# Check if the constraint is satisfied
def check_constraint(T_end_effector, T_parent, angle_threshold_deg=10):
    # Get the z-axis of the end effector frame (in world coordinates)
    z_axis_world = T_end_effector[:3, 2]
    
    # Get the positions of both joints in world coordinates
    pos_parent = T_parent[:3, 3]
    pos_end = T_end_effector[:3, 3]
    
    # Calculate the vector from parent to end effector
    vector = pos_end - pos_parent
    vector = vector / np.linalg.norm(vector)
    
    # Calculate the angle between the vector and the xy plane
    # The normal to the xy plane is [0, 0, 1]
    xy_normal = np.array([0, 0, 1])
    
    # The angle between the vector and the xy plane is 90° - angle between vector and normal
    angle_with_normal = np.arccos(np.dot(vector, xy_normal))
    angle_with_plane = np.pi/2 - angle_with_normal
    
    # Convert to degrees
    angle_with_plane_deg = np.abs(np.degrees(angle_with_plane))
    
    # Check if the angle is within the threshold
    return angle_with_plane_deg <= angle_threshold_deg

# Main function
def visualize_reachable_workspace(urdf_file, end_effector_link='fer_link8', parent_link='fer_link7', num_samples=10000, angle_threshold_deg=10):
    # Parse URDF file
    joint_limits = parse_urdf(urdf_file)
    
    # Sample joint configurations
    joint_configs = sample_joint_configurations(joint_limits, num_samples)
    
    # Calculate forward kinematics for each configuration
    valid_points = []
    manipulability_values = []
    distance_from_limits = []
    
    for joint_values in joint_configs:
        # Calculate end effector transformation
        T_end_effector = forward_kinematics(joint_limits, joint_values, end_effector_link)
        
        # Calculate parent link transformation
        T_parent = forward_kinematics(joint_limits, joint_values, parent_link)
        
        # Check if constraint is satisfied
        if check_constraint(T_end_effector, T_parent, angle_threshold_deg):
            # If satisfied, add end effector position to valid points
            valid_points.append(T_end_effector[:3, 3])
            
            # Calculate manipulability (simplified version)
            manipulability = calculate_manipulability(joint_limits, joint_values)
            manipulability_values.append(manipulability)
            
            # Calculate distance from joint limits
            distance = calculate_distance_from_limits(joint_limits, joint_values)
            distance_from_limits.append(distance)
    
    valid_points = np.array(valid_points)
    manipulability_values = np.array(manipulability_values)
    distance_from_limits = np.array(distance_from_limits)
    
    # Visualize points
    fig = plt.figure(figsize=(12, 10))
    
    # Create two subplots side by side
    ax1 = fig.add_subplot(121, projection='3d')
    ax2 = fig.add_subplot(122, projection='3d')
    
    # Normalize manipulability values for color mapping
    if len(manipulability_values) > 0:
        norm_manip = (manipulability_values - np.min(manipulability_values)) / (np.max(manipulability_values) - np.min(manipulability_values) + 1e-10)
        
        # Adjust alpha based on manipulability (higher manipulability = more opaque)
        alpha_values = 0.1 + norm_manip * 0.9  # Scale from 0.1 to 1.0
        
        # Plot points colored by manipulability with varying transparency
        scatter1 = ax1.scatter(valid_points[:, 0], valid_points[:, 1], valid_points[:, 2], 
                   c=norm_manip, cmap='viridis', marker='o', alpha=alpha_values, s=15)
        
        # Add a colorbar
        cbar1 = plt.colorbar(scatter1, ax=ax1, pad=0.1)
        cbar1.set_label('Manipulability (normalized)')
    else:
        ax1.text(0, 0, 0, "No valid points found", fontsize=12)
    
    # Normalize distance from limits
    if len(distance_from_limits) > 0:
        norm_dist = (distance_from_limits - np.min(distance_from_limits)) / (np.max(distance_from_limits) - np.min(distance_from_limits) + 1e-10)
        
        # Adjust alpha based on distance from limits (farther from limits = more opaque)
        alpha_values = 0.1 + norm_dist * 0.9  # Scale from 0.1 to 1.0
        
        # Plot points colored by distance from limits with varying transparency
        scatter2 = ax2.scatter(valid_points[:, 0], valid_points[:, 1], valid_points[:, 2], 
                   c=norm_dist, cmap='plasma', marker='o', alpha=alpha_values, s=15)
        
        # Add a colorbar
        cbar2 = plt.colorbar(scatter2, ax=ax2, pad=0.1)
        cbar2.set_label('Distance from Joint Limits (normalized)')
    else:
        ax2.text(0, 0, 0, "No valid points found", fontsize=12)
    
    # Try to compute convex hull for visualization
    try:
        if len(valid_points) > 4:  # Need at least 4 points for 3D hull
            hull = ConvexHull(valid_points)
            for simplex in hull.simplices:
                ax1.plot3D(valid_points[simplex, 0], valid_points[simplex, 1], 
                          valid_points[simplex, 2], 'r-', alpha=0.3)
                ax2.plot3D(valid_points[simplex, 0], valid_points[simplex, 1], 
                          valid_points[simplex, 2], 'r-', alpha=0.3)
    except Exception as e:
        print(f"Could not compute convex hull: {e}")
    
    # Plot the robot base
    ax1.scatter([0], [0], [0], c='r', marker='o', s=100, label='Robot Base')
    ax2.scatter([0], [0], [0], c='r', marker='o', s=100, label='Robot Base')
    
    # Set labels and titles
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_zlabel('Z (m)')
    ax1.set_title(f'Workspace Colored by Manipulability')
    
    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Y (m)')
    ax2.set_zlabel('Z (m)')
    ax2.set_title(f'Workspace Colored by Joint Limit Margin')
    
    # Set overall figure title
    plt.suptitle(f'Reachable Workspace with End Effector Line within {angle_threshold_deg}° of XY Plane', fontsize=16)
    
    # Make axis limits equal for better visualization
    if len(valid_points) > 0:
        max_range = np.max([
            np.max(valid_points[:, 0]) - np.min(valid_points[:, 0]),
            np.max(valid_points[:, 1]) - np.min(valid_points[:, 1]),
            np.max(valid_points[:, 2]) - np.min(valid_points[:, 2])
        ])
        
        mid_x = (np.max(valid_points[:, 0]) + np.min(valid_points[:, 0])) / 2
        mid_y = (np.max(valid_points[:, 1]) + np.min(valid_points[:, 1])) / 2
        mid_z = (np.max(valid_points[:, 2]) + np.min(valid_points[:, 2])) / 2
        
        ax1.set_xlim(mid_x - max_range/2, mid_x + max_range/2)
        ax1.set_ylim(mid_y - max_range/2, mid_y + max_range/2)
        ax1.set_zlim(mid_z - max_range/2, mid_z + max_range/2)
        
        ax2.set_xlim(mid_x - max_range/2, mid_x + max_range/2)
        ax2.set_ylim(mid_y - max_range/2, mid_y + max_range/2)
        ax2.set_zlim(mid_z - max_range/2, mid_z + max_range/2)
    else:
        # Default limits if no valid points
        for ax in [ax1, ax2]:
            ax.set_xlim(-1, 1)
            ax.set_ylim(-1, 1)
            ax.set_zlim(-1, 1)
    
    # Add legends
    ax1.legend()
    ax2.legend()
    
    # Show plot
    plt.tight_layout()
    plt.show()
    
    # Print statistics
    print(f"Total sampled configurations: {num_samples}")
    print(f"Valid configurations: {len(valid_points)}")
    print(f"Percentage valid: {len(valid_points) / num_samples * 100:.2f}%")
    
    if len(valid_points) > 0:
        # Calculate workspace metrics
        x_range = np.max(valid_points[:, 0]) - np.min(valid_points[:, 0])
        y_range = np.max(valid_points[:, 1]) - np.min(valid_points[:, 1])
        z_range = np.max(valid_points[:, 2]) - np.min(valid_points[:, 2])
        
        print(f"\nWorkspace dimensions:")
        print(f"X range: {x_range:.4f} m")
        print(f"Y range: {y_range:.4f} m")
        print(f"Z range: {z_range:.4f} m")
        
        if len(manipulability_values) > 0:
            print(f"\nManipulability statistics:")
            print(f"Min manipulability: {np.min(manipulability_values):.4f}")
            print(f"Max manipulability: {np.max(manipulability_values):.4f}")
            print(f"Avg manipulability: {np.mean(manipulability_values):.4f}")
        
        if len(distance_from_limits) > 0:
            print(f"\nJoint limit distance statistics:")
            print(f"Min distance from limits: {np.min(distance_from_limits):.4f}")
            print(f"Max distance from limits: {np.max(distance_from_limits):.4f}")
            print(f"Avg distance from limits: {np.mean(distance_from_limits):.4f}")
    
    # Save the figure if needed
    # plt.savefig('robot_workspace.png', dpi=300, bbox_inches='tight')
    
    # Show plot
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)  # Make room for suptitle
    plt.show()

# Usage example
if __name__ == "__main__":
    urdf_file = "test/fer.urdf"  # Path to your URDF file
    visualize_reachable_workspace(
        urdf_file,
        end_effector_link='fer_link8',
        parent_link='fer_link7',
        num_samples=20000,  # Increased number of samples
        angle_threshold_deg=10
    )