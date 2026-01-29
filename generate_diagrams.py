"""
Generate Architecture and Data Processing Diagrams for ASL Detector
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

def create_architecture_diagram():
    """Create system architecture diagram"""
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Title
    ax.text(5, 9.5, 'ASL Detector System Architecture', 
            ha='center', va='top', fontsize=18, fontweight='bold')
    
    # Define colors
    input_color = '#E3F2FD'
    process_color = '#FFF9C4'
    model_color = '#C8E6C9'
    output_color = '#FFCCBC'
    
    # Layer 1: Input Layer
    ax.add_patch(FancyBboxPatch((0.5, 7.5), 2, 1, 
                                boxstyle="round,pad=0.1", 
                                facecolor=input_color, edgecolor='black', linewidth=2))
    ax.text(1.5, 8, 'Camera Input\n(Webcam)', ha='center', va='center', fontsize=10, fontweight='bold')
    
    # Layer 2: MediaPipe Processing
    ax.add_patch(FancyBboxPatch((3.5, 7.5), 2.5, 1, 
                                boxstyle="round,pad=0.1", 
                                facecolor=process_color, edgecolor='black', linewidth=2))
    ax.text(4.75, 8, 'MediaPipe\nHand Detection', ha='center', va='center', fontsize=10, fontweight='bold')
    
    # Arrow 1->2
    arrow1 = FancyArrowPatch((2.5, 8), (3.5, 8),
                            arrowstyle='->', mutation_scale=20, linewidth=2, color='black')
    ax.add_patch(arrow1)
    
    # Layer 3: Feature Extraction
    ax.add_patch(FancyBboxPatch((7, 7.5), 2.5, 1, 
                                boxstyle="round,pad=0.1", 
                                facecolor=process_color, edgecolor='black', linewidth=2))
    ax.text(8.25, 8, 'Landmark Extraction\n(21 keypoints × 3D)', ha='center', va='center', fontsize=9, fontweight='bold')
    
    # Arrow 2->3
    arrow2 = FancyArrowPatch((6, 8), (7, 8),
                            arrowstyle='->', mutation_scale=20, linewidth=2, color='black')
    ax.add_patch(arrow2)
    
    # Layer 4: Data Collection (Optional Branch)
    ax.add_patch(FancyBboxPatch((0.5, 5.8), 2, 0.8, 
                                boxstyle="round,pad=0.1", 
                                facecolor='#E1BEE7', edgecolor='black', linewidth=1.5, linestyle='--'))
    ax.text(1.5, 6.2, 'Data Collection\n(collectdata.py)', ha='center', va='center', fontsize=8)
    
    # Layer 5: Training Pipeline (Optional Branch)
    ax.add_patch(FancyBboxPatch((0.5, 4.5), 2, 0.8, 
                                boxstyle="round,pad=0.1", 
                                facecolor='#E1BEE7', edgecolor='black', linewidth=1.5, linestyle='--'))
    ax.text(1.5, 4.9, 'Model Training\n(trainmodel.py)', ha='center', va='center', fontsize=8)
    
    # Arrow Collection->Training
    arrow_train = FancyArrowPatch((1.5, 5.8), (1.5, 5.3),
                                 arrowstyle='->', mutation_scale=15, linewidth=1.5, 
                                 color='purple', linestyle='--')
    ax.add_patch(arrow_train)
    
    # Main Model Box
    ax.add_patch(FancyBboxPatch((3.5, 5), 3, 2, 
                                boxstyle="round,pad=0.15", 
                                facecolor=model_color, edgecolor='darkgreen', linewidth=3))
    ax.text(5, 6.5, 'Bidirectional LSTM + Attention', ha='center', va='center', 
            fontsize=12, fontweight='bold')
    ax.text(5, 6, 'Model: best_model_improved.h5', ha='center', va='center', fontsize=9)
    ax.text(5, 5.6, '26 ASL Sign Classes (A-Z)', ha='center', va='center', fontsize=9)
    ax.text(5, 5.2, 'Input: Sequence of landmarks', ha='center', va='center', fontsize=8, style='italic')
    
    # Arrow from Landmarks to Model (improved path)
    arrow3 = FancyArrowPatch((7, 7.8), (6.5, 6.8),
                            arrowstyle='->', mutation_scale=20, linewidth=2.5, 
                            color='darkblue')
    ax.add_patch(arrow3)
    
    # Arrow from Training to Model
    arrow_model = FancyArrowPatch((2.5, 4.9), (3.5, 6),
                                 arrowstyle='->', mutation_scale=15, linewidth=1.5, 
                                 color='purple', linestyle='--',
                                 connectionstyle="arc3,rad=.3")
    ax.add_patch(arrow_model)
    
    # Layer 6: Prediction Output
    ax.add_patch(FancyBboxPatch((3.5, 3), 3, 1, 
                                boxstyle="round,pad=0.1", 
                                facecolor=output_color, edgecolor='black', linewidth=2))
    ax.text(5, 3.5, 'Sign Prediction\n(A-Z with confidence)', ha='center', va='center', 
            fontsize=10, fontweight='bold')
    
    # Arrow Model->Prediction
    arrow4 = FancyArrowPatch((5, 5), (5, 4),
                            arrowstyle='->', mutation_scale=20, linewidth=2, color='black')
    ax.add_patch(arrow4)
    
    # Layer 7: Visualization
    ax.add_patch(FancyBboxPatch((7.5, 3), 2, 1, 
                                boxstyle="round,pad=0.1", 
                                facecolor=output_color, edgecolor='black', linewidth=2))
    ax.text(8.5, 3.5, 'Display\n(OpenCV Window)', ha='center', va='center', 
            fontsize=10, fontweight='bold')
    
    # Arrow Prediction->Display
    arrow5 = FancyArrowPatch((6.5, 3.5), (7.5, 3.5),
                            arrowstyle='->', mutation_scale=20, linewidth=2, color='black')
    ax.add_patch(arrow5)
    
    # Layer 8: Text-to-Sign (Additional Feature)
    ax.add_patch(FancyBboxPatch((0.5, 1.5), 2, 0.8, 
                                boxstyle="round,pad=0.1", 
                                facecolor='#B3E5FC', edgecolor='black', linewidth=1.5))
    ax.text(1.5, 1.9, 'Text-to-Sign\n(text_to_sign.py)', ha='center', va='center', fontsize=8)
    
    # Arrow from Text-to-Sign to Display
    arrow_text_sign = FancyArrowPatch((2.5, 1.9), (7.5, 3.3),
                                      arrowstyle='->', mutation_scale=15, linewidth=1.5, 
                                      color='teal', linestyle='--',
                                      connectionstyle="arc3,rad=.2")
    ax.add_patch(arrow_text_sign)
    
    # Evaluation Module
    ax.add_patch(FancyBboxPatch((7.5, 1.5), 2, 0.8, 
                                boxstyle="round,pad=0.1", 
                                facecolor='#B3E5FC', edgecolor='black', linewidth=1.5))
    ax.text(8.5, 1.9, 'Model Evaluation\n(evaluate_model.py)', ha='center', va='center', fontsize=8)
    
    # Add legend
    legend_elements = [
        mpatches.Patch(facecolor=input_color, edgecolor='black', label='Input'),
        mpatches.Patch(facecolor=process_color, edgecolor='black', label='Processing'),
        mpatches.Patch(facecolor=model_color, edgecolor='black', label='ML Model'),
        mpatches.Patch(facecolor=output_color, edgecolor='black', label='Output'),
        mpatches.Patch(facecolor='#E1BEE7', edgecolor='black', label='Training Pipeline')
    ]
    ax.legend(handles=legend_elements, loc='lower center', ncol=5, 
              bbox_to_anchor=(0.5, -0.05), frameon=True)
    
    plt.tight_layout()
    plt.savefig('architecture_diagram.png', dpi=300, bbox_inches='tight')
    print("✓ Architecture diagram saved as 'architecture_diagram.png'")
    plt.close()


def create_data_processing_diagram():
    """Create data processing flow diagram"""
    fig, ax = plt.subplots(1, 1, figsize=(12, 14))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 14)
    ax.axis('off')
    
    # Title
    ax.text(5, 13.5, 'ASL Detector Data Processing Pipeline', 
            ha='center', va='top', fontsize=18, fontweight='bold')
    
    y_pos = 12.5
    step_height = 1.2
    
    # Step 1: Data Collection
    ax.add_patch(FancyBboxPatch((1, y_pos), 8, 1, 
                                boxstyle="round,pad=0.1", 
                                facecolor='#BBDEFB', edgecolor='black', linewidth=2))
    ax.text(5, y_pos+0.5, '1. Data Collection (collectdata.py)', 
            ha='center', va='center', fontsize=11, fontweight='bold')
    ax.text(5, y_pos+0.15, 'Capture video frames → MediaPipe hand detection → Save landmarks',
            ha='center', va='center', fontsize=8)
    
    y_pos -= step_height
    ax.annotate('', xy=(5, y_pos+0.9), xytext=(5, y_pos+1.1),
                arrowprops=dict(arrowstyle='->', lw=2, color='black'))
    
    # Step 2: Data Storage
    ax.add_patch(FancyBboxPatch((1, y_pos), 8, 1, 
                                boxstyle="round,pad=0.1", 
                                facecolor='#C5CAE9', edgecolor='black', linewidth=2))
    ax.text(5, y_pos+0.5, '2. Data Storage Structure', 
            ha='center', va='center', fontsize=11, fontweight='bold')
    ax.text(5, y_pos+0.15, 'MP_Data/{sign_class}/{sequence_number}/{frame_number}.npy',
            ha='center', va='center', fontsize=8, family='monospace')
    ax.text(5, y_pos-0.15, '26 classes × ~80 sequences × 30 frames = ~62K samples',
            ha='center', va='center', fontsize=8, style='italic')
    
    y_pos -= step_height
    ax.annotate('', xy=(5, y_pos+0.9), xytext=(5, y_pos+1.1),
                arrowprops=dict(arrowstyle='->', lw=2, color='black'))
    
    # Step 3: Data Loading
    ax.add_patch(FancyBboxPatch((1, y_pos), 8, 1, 
                                boxstyle="round,pad=0.1", 
                                facecolor='#C8E6C9', edgecolor='black', linewidth=2))
    ax.text(5, y_pos+0.5, '3. Data Loading & Preparation (trainmodel.py)', 
            ha='center', va='center', fontsize=11, fontweight='bold')
    ax.text(5, y_pos+0.15, 'Load .npy files → Create sequences → Assign labels',
            ha='center', va='center', fontsize=8)
    ax.text(5, y_pos-0.15, 'Shape: (samples, 30 frames, 63 features)',
            ha='center', va='center', fontsize=8, family='monospace')
    
    y_pos -= step_height
    ax.annotate('', xy=(5, y_pos+0.9), xytext=(5, y_pos+1.1),
                arrowprops=dict(arrowstyle='->', lw=2, color='black'))
    
    # Step 4: Feature Engineering
    ax.add_patch(FancyBboxPatch((1, y_pos), 8, 1.2, 
                                boxstyle="round,pad=0.1", 
                                facecolor='#FFF9C4', edgecolor='black', linewidth=2))
    ax.text(5, y_pos+0.8, '4. Feature Extraction', 
            ha='center', va='center', fontsize=11, fontweight='bold')
    ax.text(5, y_pos+0.45, 'MediaPipe Landmarks: 21 hand keypoints',
            ha='center', va='center', fontsize=8)
    ax.text(5, y_pos+0.2, 'Features per frame: x, y, z coordinates = 63 values',
            ha='center', va='center', fontsize=8)
    ax.text(5, y_pos-0.05, 'Normalized & scaled for model input',
            ha='center', va='center', fontsize=8, style='italic')
    
    y_pos -= 1.4
    ax.annotate('', xy=(5, y_pos+0.9), xytext=(5, y_pos+1.1),
                arrowprops=dict(arrowstyle='->', lw=2, color='black'))
    
    # Step 5: Data Split
    ax.add_patch(FancyBboxPatch((1, y_pos), 8, 1, 
                                boxstyle="round,pad=0.1", 
                                facecolor='#E1BEE7', edgecolor='black', linewidth=2))
    ax.text(5, y_pos+0.5, '5. Train/Test Split', 
            ha='center', va='center', fontsize=11, fontweight='bold')
    ax.text(5, y_pos+0.15, 'Training: 80% | Testing: 20% (stratified split)',
            ha='center', va='center', fontsize=8)
    
    y_pos -= step_height
    ax.annotate('', xy=(5, y_pos+0.9), xytext=(5, y_pos+1.1),
                arrowprops=dict(arrowstyle='->', lw=2, color='black'))
    
    # Step 6: Model Architecture
    ax.add_patch(FancyBboxPatch((1, y_pos-0.8), 8, 2.1, 
                                boxstyle="round,pad=0.15", 
                                facecolor='#A5D6A7', edgecolor='darkgreen', linewidth=3))
    ax.text(5, y_pos+0.9, '6. Bidirectional LSTM + Attention', 
            ha='center', va='center', fontsize=12, fontweight='bold')
    ax.text(5, y_pos+0.55, 'Layer 1: Bidirectional LSTM(128) + Dropout(0.4)', 
            ha='center', va='center', fontsize=8, family='monospace')
    ax.text(5, y_pos+0.35, 'Layer 2: Bidirectional LSTM(160) + Dropout(0.4)', 
            ha='center', va='center', fontsize=8, family='monospace')
    ax.text(5, y_pos+0.15, 'Layer 3: Bidirectional LSTM(128) + Dropout(0.4)', 
            ha='center', va='center', fontsize=8, family='monospace')
    ax.text(5, y_pos-0.05, 'Attention Layer: Focus on key frames', 
            ha='center', va='center', fontsize=8, family='monospace')
    ax.text(5, y_pos-0.25, 'Dense Layers: 256 → 128 → 26 (Softmax)', 
            ha='center', va='center', fontsize=8, family='monospace')
    
    y_pos -= 2.3
    ax.annotate('', xy=(5, y_pos+0.9), xytext=(5, y_pos+1.1),
                arrowprops=dict(arrowstyle='->', lw=2, color='black'))
    
    # Step 7: Training
    ax.add_patch(FancyBboxPatch((1, y_pos), 8, 1, 
                                boxstyle="round,pad=0.1", 
                                facecolor='#FFCCBC', edgecolor='black', linewidth=2))
    ax.text(5, y_pos+0.5, '7. Model Training', 
            ha='center', va='center', fontsize=11, fontweight='bold')
    ax.text(5, y_pos+0.15, 'Adam optimizer, categorical crossentropy loss, early stopping',
            ha='center', va='center', fontsize=8)
    ax.text(5, y_pos-0.15, 'Save best model: best_model_improved.h5',
            ha='center', va='center', fontsize=8, style='italic')
    
    y_pos -= step_height
    ax.annotate('', xy=(5, y_pos+0.9), xytext=(5, y_pos+1.1),
                arrowprops=dict(arrowstyle='->', lw=2, color='black'))
    
    # Step 8: Evaluation
    ax.add_patch(FancyBboxPatch((1, y_pos), 8, 1, 
                                boxstyle="round,pad=0.1", 
                                facecolor='#F8BBD0', edgecolor='black', linewidth=2))
    ax.text(5, y_pos+0.5, '8. Model Evaluation (evaluate_model.py)', 
            ha='center', va='center', fontsize=11, fontweight='bold')
    ax.text(5, y_pos+0.15, 'Metrics: Accuracy, Precision, Recall, F1-Score',
            ha='center', va='center', fontsize=8)
    ax.text(5, y_pos-0.15, 'Outputs: Classification reports, confusion matrices, PR curves',
            ha='center', va='center', fontsize=8, style='italic')
    
    y_pos -= step_height
    ax.annotate('', xy=(5, y_pos+0.9), xytext=(5, y_pos+1.1),
                arrowprops=dict(arrowstyle='->', lw=2, color='black'))
    
    # Step 9: Real-time Inference
    ax.add_patch(FancyBboxPatch((1, y_pos), 8, 1.2, 
                                boxstyle="round,pad=0.1", 
                                facecolor='#B2DFDB', edgecolor='black', linewidth=2))
    ax.text(5, y_pos+0.8, '9. Real-time Prediction (app.py)', 
            ha='center', va='center', fontsize=11, fontweight='bold')
    ax.text(5, y_pos+0.45, 'Webcam → Hand detection → Sequence buffer (30 frames)',
            ha='center', va='center', fontsize=8)
    ax.text(5, y_pos+0.2, 'Model prediction → Display sign + confidence',
            ha='center', va='center', fontsize=8)
    ax.text(5, y_pos-0.05, 'Real-time visualization with OpenCV',
            ha='center', va='center', fontsize=8, style='italic')
    
    # Add data flow annotations
    ax.text(9.5, 12.5, '📹', fontsize=20)
    ax.text(9.5, 10.1, '💾', fontsize=20)
    ax.text(9.5, 7.7, '🔄', fontsize=20)
    ax.text(9.5, 4.5, '🧠', fontsize=20)
    ax.text(9.5, 1.5, '📊', fontsize=20)
    
    plt.tight_layout()
    plt.savefig('data_processing_diagram.png', dpi=300, bbox_inches='tight')
    print("✓ Data processing diagram saved as 'data_processing_diagram.png'")
    plt.close()


def create_mermaid_diagrams():
    """Generate Mermaid.js code for GitHub README"""
    architecture_mermaid = """
## Architecture Diagram (Mermaid)

```mermaid
graph TB
    A[Camera Input] --> B[MediaPipe Hand Detection]
    B --> C[Landmark Extraction<br/>21 keypoints × 3D]
    C --> D[LSTM Neural Network<br/>best_model_improved.h5]
    D --> E[Sign Prediction A-Z]
    E --> F[Display Output]
    
    G[Data Collection] -.-> H[Training Data<br/>MP_Data/]
    H -.-> I[Model Training]
    I -.-> D
    
    J[Text-to-Sign] -.-> K[Sign Lookup]
    L[Model Evaluation] -.-> D
    
    style D fill:#c8e6c9,stroke:#388e3c,stroke-width:3px
    style A fill:#e3f2fd
    style F fill:#ffccbc
    style I fill:#e1bee7
```
"""
    
    data_flow_mermaid = """
## Data Processing Flow (Mermaid)

```mermaid
flowchart TD
    A[📹 Video Capture] --> B[MediaPipe Processing]
    B --> C[Extract 63 Features<br/>x,y,z × 21 landmarks]
    C --> D[💾 Save to MP_Data/]
    D --> E[Load Training Data<br/>~62K samples]
    E --> F[Train/Test Split<br/>80/20]
    F --> G[🧠 LSTM Model<br/>3 layers + Dense]
    G --> H[Training with<br/>Early Stopping]
    H --> I[📊 Save Best Model]
    I --> J[Evaluation Metrics]
    J --> K[Real-time Inference]
    
    style G fill:#a5d6a7,stroke:#2e7d32,stroke-width:3px
    style A fill:#bbdefb
    style I fill:#ffccbc
```
"""
    
    with open('diagrams_mermaid.md', 'w') as f:
        f.write("# ASL Detector Diagrams (Mermaid.js)\n\n")
        f.write(architecture_mermaid)
        f.write("\n\n")
        f.write(data_flow_mermaid)
    
    print("✓ Mermaid diagrams saved as 'diagrams_mermaid.md'")
    print("  You can copy this into your GitHub README.md")


if __name__ == "__main__":
    print("Generating ASL Detector Diagrams...\n")
    
    create_architecture_diagram()
    create_data_processing_diagram()
    create_mermaid_diagrams()
    
    print("\n" + "="*60)
    print("✅ All diagrams generated successfully!")
    print("="*60)
    print("\nGenerated files:")
    print("  1. architecture_diagram.png - System architecture")
    print("  2. data_processing_diagram.png - Data flow pipeline")
    print("  3. diagrams_mermaid.md - Mermaid code for GitHub")
    print("\nYou can now:")
    print("  • View the PNG images")
    print("  • Add Mermaid code to your README.md")
    print("  • Use these in presentations or documentation")
