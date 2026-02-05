import numpy as np

class KeypointNormalizer:
    @staticmethod
    def normalize(keypoints):
        """
        Normalize hand landmarks to be scale and position invariant.
        
        Args:
            keypoints: 1D array of 63 values (21 landmarks * 3 coords [x, y, z])
            
        Returns:
            Normalized 1D array of 63 values centered at (0,0) and scaled to unit box.
        """
        # Reshape to (21, 3)
        landmarks = keypoints.reshape(-1, 3)
        
        # 1. Center around the wrist (landmark 0) or mean
        # Using wrist (index 0) as origin is efficient and robust
        center = landmarks[0]
        centered = landmarks - center
        
        # 2. Scale by the maximum absolute distance from the origin
        # This fits the hand into a [-1, 1] cube while preserving aspect ratio
        max_dist = np.max(np.abs(centered))
        
        # Avoid division by zero
        if max_dist < 1e-6:
            max_dist = 1.0
            
        normalized = centered / max_dist
        
        return normalized.flatten()

    @staticmethod
    def normalize_sequence(sequence):
        """
        Normalize a sequence of frames.
        
        Args:
            sequence: Array of shape (seq_len, 63)
            
        Returns:
            Normalized sequence of shape (seq_len, 63)
        """
        return np.array([KeypointNormalizer.normalize(frame) for frame in sequence])
    
    @staticmethod
    def normalize_two_hands(keypoints):
        """
        Normalize two-hand landmarks to be scale and position invariant.
        Each hand is normalized independently with its own wrist as origin.
        
        Args:
            keypoints: 1D array of 126 values (2 hands * 21 landmarks * 3 coords)
                       First 63 values = left hand, next 63 = right hand
            
        Returns:
            Normalized 1D array of 126 values.
        """
        left_hand = keypoints[:63]
        right_hand = keypoints[63:126]
        
        # Normalize each hand independently (handles zeros for missing hands)
        left_normalized = KeypointNormalizer.normalize(left_hand) if np.any(left_hand) else left_hand
        right_normalized = KeypointNormalizer.normalize(right_hand) if np.any(right_hand) else right_hand
        
        return np.concatenate([left_normalized, right_normalized])
    
    @staticmethod
    def normalize_sequence_two_hands(sequence):
        """
        Normalize a sequence of two-hand frames.
        
        Args:
            sequence: Array of shape (seq_len, 126)
            
        Returns:
            Normalized sequence of shape (seq_len, 126)
        """
        return np.array([KeypointNormalizer.normalize_two_hands(frame) for frame in sequence])