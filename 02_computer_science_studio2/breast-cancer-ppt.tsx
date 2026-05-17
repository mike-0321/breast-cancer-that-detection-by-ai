import { useState } from "react";

const slides = [
  {
    id: 0,
    type: "title",
    title: "Breast Cancer Classification\nwith Mask-Guided Attention",
    subtitle: "DenseNet121  ·  Spatial Attention  ·  Grad-CAM Explainability",
    content: {
      author: "Presenter: Mike",
      details: ["Deep Learning", "Medical Imaging", "Computer-Aided Diagnosis"]
    }
  },
  {
    id: 1,
    title: "Background & Motivation",
    type: "two_col",
    content: {
      left: {
        heading: "Clinical Problem",
        color: "#3B82F6",
        points: [
          "Breast cancer is the most common malignancy in women worldwide",
          "Mammography is the primary screening modality",
          "Manual reading: time-consuming, subjective, 10–30% miss rate",
          "Early detection raises 5-year survival to >90%"
        ]
      },
      right: {
        heading: "Our Objective",
        color: "#8B5CF6",
        points: [
          "Build an automatic benign vs malignant classifier",
          "Guide model attention using breast segmentation masks",
          "Suppress background noise & pectoral muscle artifacts",
          "Provide Grad-CAM explainability for clinical trust"
        ]
      }
    }
  },
  {
    id: 2,
    title: "Dataset Overview",
    type: "stats",
    content: {
      stats: [
        { label: "Training Set", value: "~Thousands", icon: "📁" },
        { label: "Validation Set", value: "552", icon: "📊" },
        { label: "Task", value: "Binary", icon: "🏷️" },
        { label: "Format", value: "DICOM/PNG", icon: "🖼️" }
      ],
      notes: [
        "Class 0 = Benign,  Class 1 = Malignant",
        "Stored on Google Drive, trained on Google Colab (GPU)",
        "White-background images filtered out during preprocessing",
        "Weighted cross-entropy loss to handle class imbalance"
      ]
    }
  },
  {
    id: 3,
    title: "Preprocessing Pipeline",
    type: "pipeline",
    content: {
      steps: [
        { num: "1", title: "Grayscale Loading", desc: "Read as float32, normalize to [0, 1]", color: "#3B82F6" },
        { num: "2", title: "White-BG Filter", desc: "Detect and skip non-mammogram white-background images", color: "#8B5CF6" },
        { num: "3", title: "Breast Mask Generation", desc: "Otsu threshold + morphological ops + largest connected component", color: "#EC4899" },
        { num: "4", title: "Pectoral Muscle Removal", desc: "Locate bright triangular region in upper corner (MLO views)", color: "#F59E0B" },
        { num: "5", title: "Crop & Pad to 224×224", desc: "Crop to mask bounding box → pad to square → resize (no stretching)", color: "#10B981" }
      ],
      keyPoint: "Single-channel grayscale throughout — no artificial 3-channel duplication"
    }
  },
  {
    id: 4,
    title: "Model Architecture",
    type: "architecture",
    content: {
      layers: [
        { name: "Input Layer", detail: "Single-channel grayscale (1, 224, 224)", color: "#6366F1" },
        { name: "DenseNet121 Backbone", detail: "MONAI implementation, in_channels=1 → 1024-dim feature maps", color: "#3B82F6" },
        { name: "SE Channel Attention", detail: "Squeeze-and-Excitation (reduction=16) — learns which channels matter", color: "#8B5CF6" },
        { name: "Mask-Guided Spatial Attention", detail: "1×1 Conv → Sigmoid × downsampled breast mask — constrains WHERE to attend", color: "#EC4899" },
        { name: "Classification Head", detail: "GAP → Dropout(0.4) → FC(256) → ReLU → Dropout(0.2) → FC(2)", color: "#10B981" }
      ],
      params: "Total parameters: ~7.0M",
      highlight: "Core Innovation: breast mask constrains attention to clinically relevant regions only"
    }
  },
  {
    id: 5,
    title: "Training Strategy",
    type: "two_col",
    content: {
      left: {
        heading: "Hyperparameters",
        color: "#3B82F6",
        points: [
          "Max epochs: 30",
          "Learning rate: 1e-4 (Adam)",
          "Weight decay: 1e-4",
          "LR schedule: Cosine Annealing → 1e-6",
          "Early stopping: patience = 10",
          "Batch size: 16"
        ]
      },
      right: {
        heading: "Data Augmentation (MONAI)",
        color: "#8B5CF6",
        points: [
          "Random horizontal / vertical flip (p=0.5)",
          "Random 90° rotation (p=0.5)",
          "Random zoom 0.85–1.15 (p=0.3)",
          "Gaussian noise injection (p=0.2)",
          "Random contrast adjustment (p=0.2)",
          "Weighted cross-entropy for class imbalance"
        ]
      }
    }
  },
  {
    id: 6,
    title: "Training Process",
    type: "training",
    content: {
      charts: [
        { title: "Loss Curves", desc: "Training loss decreases steadily; validation loss stabilizes", icon: "📉" },
        { title: "Validation Accuracy", desc: "Gradually increases and plateaus; best model saved automatically", icon: "📈" },
        { title: "Learning Rate", desc: "Cosine Annealing provides smooth decay, preventing late oscillations", icon: "⚙️" }
      ],
      note: "Early stopping ensures the model halts before overfitting"
    }
  },
  {
    id: 7,
    title: "Evaluation Results",
    type: "results",
    content: {
      metrics: [
        { name: "Accuracy", value: "—", desc: "Overall correct rate" },
        { name: "Precision", value: "—", desc: "True pos among predicted" },
        { name: "Recall", value: "—", desc: "True pos among actual" },
        { name: "F1-Score", value: "—", desc: "Harmonic mean of P & R" },
        { name: "AUC", value: "—", desc: "Area under ROC curve" }
      ],
      cmNote: "Confusion matrix shows per-class prediction distribution",
      note: "Replace \"—\" with your actual metric values after training"
    }
  },
  {
    id: 8,
    title: "Grad-CAM Explainability",
    type: "gradcam",
    content: {
      columns: [
        { title: "Original Image", desc: "Preprocessed\ngrayscale mammogram", icon: "🖼️" },
        { title: "Breast Mask", desc: "Binary mask from\nOtsu + morphology", icon: "🎭" },
        { title: "Grad-CAM Overlay", desc: "Heatmap overlaid\non original image", icon: "🔥" },
        { title: "Heatmap", desc: "Red = high attention\nBlue = low attention", icon: "🌡️" }
      ],
      findings: [
        "Correct predictions → attention concentrates on lesion area inside breast",
        "Incorrect predictions → attention scattered or at edges",
        "Mask guidance effectively suppresses background activation"
      ]
    }
  },
  {
    id: 9,
    title: "Mask-Guided vs Unguided Attention",
    type: "comparison",
    content: {
      comparisons: [
        { aspect: "Attention Range", with_mask: "Strictly within breast region", without: "May spread to background & pectoral" },
        { aspect: "Reliability", with_mask: "Reduces background artifact interference", without: "May learn diagnosis-irrelevant features" },
        { aspect: "Clinical Interpretability", with_mask: "Focus aligns with radiologist reading", without: "Focus areas may be unreasonable" },
        { aspect: "Diff Map", with_mask: "Red = mask-enhanced regions", without: "Blue = mask-suppressed regions" }
      ]
    }
  },
  {
    id: 10,
    title: "Embedding Space Analysis (t-SNE)",
    type: "embedding",
    content: {
      findings: [
        { label: "t-SNE Scatter Plot", detail: "Benign & malignant show some clustering, but significant overlap remains" },
        { label: "Density Distribution", detail: "KDE curves of both classes overlap heavily along both t-SNE dimensions" },
        { label: "Linear Separability", detail: "Logistic Regression on 1024-d embeddings: ~65.8% — limited discriminability" }
      ],
      interpretation: "The model has learned some discriminative features, but benign vs malignant representations are not yet well-separated — room for improvement."
    }
  },
  {
    id: 11,
    title: "Conclusion & Future Work",
    type: "conclusion",
    content: {
      achievements: [
        "End-to-end pipeline: preprocessing → training → evaluation → visualization",
        "Introduced mask-guided spatial attention mechanism",
        "Grad-CAM explainability with mask vs no-mask comparison",
        "t-SNE embedding analysis reveals feature space structure"
      ],
      future: [
        "Transfer learning with ImageNet-pretrained weights",
        "Contrastive learning to improve embedding separability",
        "Expand dataset (CBIS-DDSM, VinDr-Mammo)",
        "Multi-task learning: classification + lesion localization"
      ]
    }
  },
  {
    id: 12,
    title: "Thank You!",
    type: "thanks",
    content: {
      text: "Q & A",
      details: ["Questions and discussion are welcome."]
    }
  }
];

function TitleSlide({ slide }) {
  return (
    <div style={{ display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", height: "100%", textAlign: "center", padding: "0 48px" }}>
      <div style={{ fontSize: 13, color: "#94A3B8", letterSpacing: 4, marginBottom: 24, textTransform: "uppercase", fontWeight: 500 }}>Deep Learning in Medical Imaging</div>
      <h1 style={{ fontSize: 30, fontWeight: 800, lineHeight: 1.35, color: "#F1F5F9", whiteSpace: "pre-line", marginBottom: 14 }}>{slide.title}</h1>
      <div style={{ fontSize: 14, color: "#818CF8", marginBottom: 36, fontWeight: 500 }}>{slide.subtitle}</div>
      <div style={{ fontSize: 17, color: "#CBD5E1", marginBottom: 14 }}>{slide.content.author}</div>
      <div style={{ display: "flex", gap: 10 }}>
        {slide.content.details.map((d, i) => (
          <span key={i} style={{ fontSize: 11, color: "#94A3B8", background: "rgba(99,102,241,0.12)", padding: "5px 16px", borderRadius: 20, border: "1px solid rgba(99,102,241,0.2)" }}>{d}</span>
        ))}
      </div>
    </div>
  );
}

function TwoColSlide({ slide }) {
  const { left, right } = slide.content;
  return (
    <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 24, padding: "8px 20px", height: "100%" }}>
      {[left, right].map((col, ci) => (
        <div key={ci} style={{ background: "rgba(30,41,59,0.55)", borderRadius: 14, padding: "22px 24px", border: "1px solid rgba(148,163,184,0.08)", borderLeft: `3px solid ${col.color}` }}>
          <h3 style={{ fontSize: 17, fontWeight: 700, color: col.color, marginBottom: 18 }}>{col.heading}</h3>
          {col.points.map((p, i) => (
            <div key={i} style={{ display: "flex", alignItems: "flex-start", gap: 10, marginBottom: 12, fontSize: 13.5, color: "#CBD5E1", lineHeight: 1.55 }}>
              <span style={{ color: col.color, fontWeight: 700, flexShrink: 0, marginTop: 1 }}>›</span>
              <span>{p}</span>
            </div>
          ))}
        </div>
      ))}
    </div>
  );
}

function StatsSlide({ slide }) {
  const colors = ["#3B82F6", "#8B5CF6", "#10B981", "#F59E0B"];
  return (
    <div style={{ padding: "8px 20px" }}>
      <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: 14, marginBottom: 22 }}>
        {slide.content.stats.map((s, i) => (
          <div key={i} style={{ background: "rgba(30,41,59,0.6)", borderRadius: 14, padding: "20px 14px", textAlign: "center", border: "1px solid rgba(148,163,184,0.08)", borderBottom: `3px solid ${colors[i]}` }}>
            <div style={{ fontSize: 28, marginBottom: 8 }}>{s.icon}</div>
            <div style={{ fontSize: 20, fontWeight: 800, color: "#F1F5F9" }}>{s.value}</div>
            <div style={{ fontSize: 12, color: "#94A3B8", marginTop: 6 }}>{s.label}</div>
          </div>
        ))}
      </div>
      <div style={{ background: "rgba(30,41,59,0.45)", borderRadius: 14, padding: "18px 22px", border: "1px solid rgba(148,163,184,0.08)" }}>
        {slide.content.notes.map((n, i) => (
          <div key={i} style={{ fontSize: 13.5, color: "#CBD5E1", marginBottom: 9, display: "flex", gap: 10, lineHeight: 1.5 }}>
            <span style={{ color: "#6366F1", fontWeight: 700 }}>•</span>{n}
          </div>
        ))}
      </div>
    </div>
  );
}

function PipelineSlide({ slide }) {
  return (
    <div style={{ padding: "8px 20px" }}>
      <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
        {slide.content.steps.map((s, i) => (
          <div key={i} style={{ display: "flex", alignItems: "center", gap: 14 }}>
            <div style={{ width: 38, height: 38, borderRadius: "50%", background: s.color, display: "flex", alignItems: "center", justifyContent: "center", fontWeight: 800, fontSize: 15, color: "#fff", flexShrink: 0 }}>{s.num}</div>
            <div style={{ flex: 1, background: "rgba(30,41,59,0.55)", borderRadius: 10, padding: "11px 18px", borderLeft: `3px solid ${s.color}` }}>
              <div style={{ fontWeight: 700, fontSize: 14, color: "#F1F5F9", marginBottom: 2 }}>{s.title}</div>
              <div style={{ fontSize: 12, color: "#94A3B8" }}>{s.desc}</div>
            </div>
          </div>
        ))}
      </div>
      <div style={{ marginTop: 14, padding: "10px 16px", background: "rgba(16,185,129,0.08)", borderRadius: 10, border: "1px solid rgba(16,185,129,0.25)", fontSize: 12.5, color: "#6EE7B7", display: "flex", gap: 8 }}>
        <span>💡</span> {slide.content.keyPoint}
      </div>
    </div>
  );
}

function ArchitectureSlide({ slide }) {
  return (
    <div style={{ padding: "8px 20px" }}>
      <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
        {slide.content.layers.map((l, i) => (
          <div key={i}>
            <div style={{ background: "rgba(30,41,59,0.55)", borderRadius: 10, padding: "11px 18px", borderLeft: `4px solid ${l.color}` }}>
              <div style={{ fontWeight: 700, fontSize: 14, color: "#F1F5F9" }}>{l.name}</div>
              <div style={{ fontSize: 11.5, color: "#94A3B8", marginTop: 2 }}>{l.detail}</div>
            </div>
            {i < slide.content.layers.length - 1 && (
              <div style={{ textAlign: "center", color: "#475569", fontSize: 12, padding: "1px 0" }}>▼</div>
            )}
          </div>
        ))}
      </div>
      <div style={{ display: "flex", gap: 12, marginTop: 10 }}>
        <div style={{ flex: 1, padding: "9px 14px", background: "rgba(99,102,241,0.08)", borderRadius: 8, fontSize: 12, color: "#A5B4FC", border: "1px solid rgba(99,102,241,0.15)" }}>{slide.content.params}</div>
        <div style={{ flex: 2.5, padding: "9px 14px", background: "rgba(236,72,153,0.08)", borderRadius: 8, fontSize: 12, color: "#F9A8D4", border: "1px solid rgba(236,72,153,0.15)" }}>⭐ {slide.content.highlight}</div>
      </div>
    </div>
  );
}

function TrainingSlide({ slide }) {
  return (
    <div style={{ padding: "16px 20px" }}>
      <div style={{ display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: 16, marginBottom: 20 }}>
        {slide.content.charts.map((c, i) => (
          <div key={i} style={{ background: "rgba(30,41,59,0.55)", borderRadius: 14, padding: "24px 18px", textAlign: "center", border: "1px solid rgba(148,163,184,0.08)" }}>
            <div style={{ fontSize: 40, marginBottom: 12 }}>{c.icon}</div>
            <div style={{ fontWeight: 700, fontSize: 15, color: "#F1F5F9", marginBottom: 8 }}>{c.title}</div>
            <div style={{ fontSize: 12.5, color: "#94A3B8", lineHeight: 1.5 }}>{c.desc}</div>
          </div>
        ))}
      </div>
      <div style={{ padding: "11px 16px", background: "rgba(251,191,36,0.08)", borderRadius: 10, border: "1px solid rgba(251,191,36,0.2)", fontSize: 13, color: "#FCD34D", textAlign: "center" }}>
        ⚡ {slide.content.note}
      </div>
    </div>
  );
}

function ResultsSlide({ slide }) {
  const colors = ["#3B82F6", "#8B5CF6", "#EC4899", "#10B981", "#F59E0B"];
  return (
    <div style={{ padding: "8px 20px" }}>
      <div style={{ display: "grid", gridTemplateColumns: "repeat(5, 1fr)", gap: 10, marginBottom: 20 }}>
        {slide.content.metrics.map((m, i) => (
          <div key={i} style={{ background: "rgba(30,41,59,0.6)", borderRadius: 14, padding: "18px 10px", textAlign: "center", border: "1px solid rgba(148,163,184,0.08)", borderTop: `3px solid ${colors[i]}` }}>
            <div style={{ fontSize: 24, fontWeight: 800, color: colors[i], marginBottom: 6 }}>{m.value}</div>
            <div style={{ fontSize: 13, fontWeight: 700, color: "#F1F5F9", marginBottom: 4 }}>{m.name}</div>
            <div style={{ fontSize: 10.5, color: "#94A3B8" }}>{m.desc}</div>
          </div>
        ))}
      </div>
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12 }}>
        <div style={{ background: "rgba(30,41,59,0.45)", borderRadius: 12, padding: 16, border: "1px solid rgba(148,163,184,0.08)" }}>
          <div style={{ fontSize: 13, fontWeight: 700, color: "#F1F5F9", marginBottom: 6 }}>📊 Confusion Matrix</div>
          <div style={{ fontSize: 12.5, color: "#94A3B8" }}>{slide.content.cmNote}</div>
        </div>
        <div style={{ background: "rgba(251,191,36,0.06)", borderRadius: 12, padding: 16, border: "1px solid rgba(251,191,36,0.15)" }}>
          <div style={{ fontSize: 13, fontWeight: 700, color: "#FCD34D", marginBottom: 6 }}>📌 Note</div>
          <div style={{ fontSize: 12.5, color: "#94A3B8" }}>{slide.content.note}</div>
        </div>
      </div>
    </div>
  );
}

function GradcamSlide({ slide }) {
  return (
    <div style={{ padding: "8px 20px" }}>
      <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: 12, marginBottom: 18 }}>
        {slide.content.columns.map((c, i) => (
          <div key={i} style={{ background: "rgba(30,41,59,0.55)", borderRadius: 14, padding: "20px 14px", textAlign: "center", border: "1px solid rgba(148,163,184,0.08)" }}>
            <div style={{ fontSize: 32, marginBottom: 10 }}>{c.icon}</div>
            <div style={{ fontWeight: 700, fontSize: 13.5, color: "#F1F5F9", marginBottom: 6 }}>{c.title}</div>
            <div style={{ fontSize: 11.5, color: "#94A3B8", whiteSpace: "pre-line", lineHeight: 1.5 }}>{c.desc}</div>
          </div>
        ))}
      </div>
      <div style={{ background: "rgba(30,41,59,0.45)", borderRadius: 14, padding: "16px 20px", border: "1px solid rgba(148,163,184,0.08)" }}>
        <div style={{ fontSize: 14, fontWeight: 700, color: "#10B981", marginBottom: 10 }}>🔍 Key Findings</div>
        {slide.content.findings.map((f, i) => (
          <div key={i} style={{ fontSize: 13, color: "#CBD5E1", marginBottom: 7, display: "flex", gap: 8, lineHeight: 1.5 }}>
            <span style={{ color: "#10B981", flexShrink: 0 }}>✓</span>{f}
          </div>
        ))}
      </div>
    </div>
  );
}

function ComparisonSlide({ slide }) {
  return (
    <div style={{ padding: "8px 20px" }}>
      <div style={{ background: "rgba(30,41,59,0.45)", borderRadius: 14, overflow: "hidden", border: "1px solid rgba(148,163,184,0.08)" }}>
        <div style={{ display: "grid", gridTemplateColumns: "1.3fr 1.5fr 1.5fr", background: "rgba(99,102,241,0.1)", padding: "14px 20px" }}>
          <div style={{ fontSize: 13, fontWeight: 700, color: "#A5B4FC" }}>Aspect</div>
          <div style={{ fontSize: 13, fontWeight: 700, color: "#34D399" }}>✅ With Mask Guidance</div>
          <div style={{ fontSize: 13, fontWeight: 700, color: "#FB923C" }}>❌ Without Mask</div>
        </div>
        {slide.content.comparisons.map((c, i) => (
          <div key={i} style={{ display: "grid", gridTemplateColumns: "1.3fr 1.5fr 1.5fr", padding: "14px 20px", borderTop: "1px solid rgba(148,163,184,0.06)", background: i % 2 === 0 ? "transparent" : "rgba(30,41,59,0.3)" }}>
            <div style={{ fontSize: 13, fontWeight: 600, color: "#CBD5E1" }}>{c.aspect}</div>
            <div style={{ fontSize: 12.5, color: "#6EE7B7" }}>{c.with_mask}</div>
            <div style={{ fontSize: 12.5, color: "#FDBA74" }}>{c.without}</div>
          </div>
        ))}
      </div>
    </div>
  );
}

function EmbeddingSlide({ slide }) {
  return (
    <div style={{ padding: "8px 20px" }}>
      <div style={{ display: "flex", flexDirection: "column", gap: 12, marginBottom: 16 }}>
        {slide.content.findings.map((f, i) => (
          <div key={i} style={{ background: "rgba(30,41,59,0.55)", borderRadius: 12, padding: "14px 20px", borderLeft: "3px solid #6366F1" }}>
            <div style={{ fontWeight: 700, fontSize: 14.5, color: "#A5B4FC", marginBottom: 4 }}>{f.label}</div>
            <div style={{ fontSize: 13, color: "#94A3B8", lineHeight: 1.5 }}>{f.detail}</div>
          </div>
        ))}
      </div>
      <div style={{ padding: "14px 20px", background: "rgba(251,191,36,0.08)", borderRadius: 12, border: "1px solid rgba(251,191,36,0.2)" }}>
        <div style={{ fontSize: 13, fontWeight: 700, color: "#FCD34D", marginBottom: 6 }}>💡 Interpretation</div>
        <div style={{ fontSize: 13, color: "#CBD5E1", lineHeight: 1.5 }}>{slide.content.interpretation}</div>
      </div>
    </div>
  );
}

function ConclusionSlide({ slide }) {
  return (
    <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 20, padding: "8px 20px" }}>
      <div style={{ background: "rgba(16,185,129,0.06)", borderRadius: 14, padding: "22px 24px", border: "1px solid rgba(16,185,129,0.15)" }}>
        <h3 style={{ fontSize: 16, fontWeight: 700, color: "#34D399", marginBottom: 16, paddingBottom: 10, borderBottom: "2px solid rgba(16,185,129,0.2)" }}>✅ Achievements</h3>
        {slide.content.achievements.map((a, i) => (
          <div key={i} style={{ fontSize: 13, color: "#CBD5E1", marginBottom: 10, display: "flex", gap: 10, lineHeight: 1.5 }}>
            <span style={{ color: "#10B981", flexShrink: 0 }}>✓</span>{a}
          </div>
        ))}
      </div>
      <div style={{ background: "rgba(99,102,241,0.06)", borderRadius: 14, padding: "22px 24px", border: "1px solid rgba(99,102,241,0.15)" }}>
        <h3 style={{ fontSize: 16, fontWeight: 700, color: "#818CF8", marginBottom: 16, paddingBottom: 10, borderBottom: "2px solid rgba(99,102,241,0.2)" }}>🚀 Future Directions</h3>
        {slide.content.future.map((f, i) => (
          <div key={i} style={{ fontSize: 13, color: "#CBD5E1", marginBottom: 10, display: "flex", gap: 10, lineHeight: 1.5 }}>
            <span style={{ color: "#6366F1", flexShrink: 0 }}>→</span>{f}
          </div>
        ))}
      </div>
    </div>
  );
}

function ThanksSlide({ slide }) {
  return (
    <div style={{ display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", height: "100%", textAlign: "center" }}>
      <div style={{ fontSize: 52, fontWeight: 800, color: "#F1F5F9", marginBottom: 20 }}>{slide.title}</div>
      <div style={{ width: 80, height: 3, background: "linear-gradient(90deg, #6366F1, #EC4899)", borderRadius: 2, marginBottom: 24 }} />
      <div style={{ fontSize: 26, color: "#818CF8", fontWeight: 600, marginBottom: 20 }}>{slide.content.text}</div>
      {slide.content.details.map((d, i) => (
        <div key={i} style={{ fontSize: 15, color: "#94A3B8" }}>{d}</div>
      ))}
    </div>
  );
}

const renderers = {
  title: TitleSlide, two_col: TwoColSlide, stats: StatsSlide,
  pipeline: PipelineSlide, architecture: ArchitectureSlide,
  training: TrainingSlide, results: ResultsSlide, gradcam: GradcamSlide,
  comparison: ComparisonSlide, embedding: EmbeddingSlide,
  conclusion: ConclusionSlide, thanks: ThanksSlide
};

export default function App() {
  const [idx, setIdx] = useState(0);
  const slide = slides[idx];
  const Renderer = renderers[slide.type];

  const handleKey = (e) => {
    if (e.key === "ArrowRight" || e.key === " ") setIdx(i => Math.min(slides.length - 1, i + 1));
    if (e.key === "ArrowLeft") setIdx(i => Math.max(0, i - 1));
  };

  return (
    <div style={{ width: "100%", maxWidth: 900, margin: "0 auto", fontFamily: "'Inter','Segoe UI',system-ui,sans-serif" }} tabIndex={0} onKeyDown={handleKey}>
      <div style={{ background: "linear-gradient(145deg, #0F172A 0%, #1E293B 50%, #0F172A 100%)", borderRadius: 16, overflow: "hidden", boxShadow: "0 24px 64px rgba(0,0,0,0.5)", border: "1px solid rgba(148,163,184,0.08)" }}>
        {slide.type !== "title" && slide.type !== "thanks" && (
          <div style={{ padding: "22px 28px 0" }}>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 8 }}>
              <div style={{ fontSize: 10, color: "#6366F1", letterSpacing: 2, fontWeight: 600 }}>SLIDE {idx + 1} / {slides.length}</div>
              <div style={{ fontSize: 10, color: "#475569" }}>Breast Cancer Classification</div>
            </div>
            <h2 style={{ fontSize: 23, fontWeight: 800, color: "#F1F5F9", margin: 0, paddingBottom: 14, borderBottom: "1px solid rgba(148,163,184,0.08)" }}>{slide.title}</h2>
          </div>
        )}
        <div style={{ minHeight: 400, padding: slide.type === "title" || slide.type === "thanks" ? "48px 0" : "14px 0" }}>
          <Renderer slide={slide} />
        </div>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", padding: "14px 28px", borderTop: "1px solid rgba(148,163,184,0.06)", background: "rgba(15,23,42,0.4)" }}>
          <button onClick={() => setIdx(Math.max(0, idx - 1))} disabled={idx === 0}
            style={{ padding: "9px 22px", borderRadius: 8, border: "1px solid rgba(148,163,184,0.15)", background: idx === 0 ? "transparent" : "rgba(99,102,241,0.1)", color: idx === 0 ? "#334155" : "#A5B4FC", cursor: idx === 0 ? "default" : "pointer", fontSize: 13, fontWeight: 600, transition: "all 0.15s" }}>
            ← Previous
          </button>
          <div style={{ display: "flex", gap: 5 }}>
            {slides.map((_, i) => (
              <div key={i} onClick={() => setIdx(i)}
                style={{ width: i === idx ? 22 : 8, height: 8, borderRadius: 4, background: i === idx ? "#6366F1" : "rgba(148,163,184,0.15)", cursor: "pointer", transition: "all 0.2s" }} />
            ))}
          </div>
          <button onClick={() => setIdx(Math.min(slides.length - 1, idx + 1))} disabled={idx === slides.length - 1}
            style={{ padding: "9px 22px", borderRadius: 8, border: "none", background: idx === slides.length - 1 ? "rgba(148,163,184,0.08)" : "#6366F1", color: idx === slides.length - 1 ? "#334155" : "#fff", cursor: idx === slides.length - 1 ? "default" : "pointer", fontSize: 13, fontWeight: 600, transition: "all 0.15s" }}>
            Next →
          </button>
        </div>
      </div>
    </div>
  );
}
