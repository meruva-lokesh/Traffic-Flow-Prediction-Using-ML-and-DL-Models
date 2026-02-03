# Lucidchart AI Prompts for Conference Paper Diagrams

## How to Use:
1. Go to https://lucid.app/
2. Click **"Create"** → **"Lucidchart"**
3. Look for **"Generate with AI"** or **"AI Diagram"** button
4. Copy-paste each prompt below
5. AI will generate the diagram
6. Adjust colors and styling as needed

---

## 📊 PROMPT 1: Systematic Literature Review (PRISMA Flowchart)

```
Create a PRISMA systematic literature review flowchart with the following structure:

Start with "Identification" phase: Database Search from IEEE, Springer, ScienceDirect, and ACM with 450 papers.

Flow down to "Screening" phase: Title & Abstract Review of 450 papers. Add arrow to the right showing "Excluded: 280 papers (not traffic prediction, different domain)".

Flow down to "Eligibility" phase: Full-Text Assessment of 170 papers. Add arrow to the right showing "Excluded: 120 papers (no ML/DL methods, insufficient data, poor methodology)".

Flow down to "Included" phase: Final Papers for Review, 50 papers total.

Flow down to "Analysis Categories" (bold box).

Split into three parallel boxes at the bottom:
- Traditional ML: 15 papers (RF, SVM, DT)
- Deep Learning: 25 papers (CNN, LSTM, RNN)  
- Hybrid Models: 10 papers (Combined approaches)

Use colors: Light blue for main flow, light red for excluded boxes, light green for final analysis boxes. Add all connecting arrows vertically and horizontally.
```

---

## 🏗️ PROMPT 2: System Architecture Diagram

```
Create a layered system architecture diagram for traffic flow prediction with the following layers from top to bottom:

Layer 1 - Data Collection (3 boxes horizontally):
- Traffic Data Collection
- Weather Data
- Junction Info
All converge down with arrows.

Layer 2 - Data Preprocessing (single box):
Contains: Missing value handling, Outlier removal, Normalization

Layer 3 - Feature Engineering (single box):
Contains: Temporal features (Hour, Day), Interaction features, Categorical encoding

Layer 4 - Model Layer (3 boxes horizontally):
- Left: Traditional ML (Random Forest, Decision Tree, SVM)
- CENTER (HIGHLIGHT IN BOLD): 1D CNN Proposed Model (Conv1D layers, Batch Norm, Dropout)
- Right: Transfer Learning (VGG16, VGG19, ResNet50)
All converge down with arrows.

Layer 5 - Model Evaluation (single box):
Contains: Accuracy 92.80%, Precision, Recall, F1, McNemar Test

Layer 6 - Output (single box):
Traffic Prediction: Low | Medium | High | Severe

Use colors: Light blue for data layer, yellow for preprocessing, orange for feature engineering, green for models (bright green for 1D CNN), purple for evaluation, red for output. All arrows flow downward showing the pipeline.
```

---

## � PROMPT 3: Model Performance Comparison Bar Chart

```
Create a horizontal bar chart titled "Model Performance Comparison" showing accuracy results for 6 machine learning models:

The chart should display:
1. 1D CNN (Optimized): 92.20% - GOLD/YELLOW bar (longest bar, marked as "Best Model" with a red dashed vertical line at 92.20%)
2. Random Forest: 91.20% - LIGHT GREEN bar
3. Decision Tree: 86.70% - GREEN bar  
4. SVM: 86.20% - GREEN bar
5. Logistic Regression: 83.30% - GREEN bar
6. Naive Bayes: 79.90% - GREEN bar (shortest bar)

Layout: Horizontal bars extending from left (0%) to right (100%)
Y-axis: Model names (listed vertically from top to bottom as shown above)
X-axis: Accuracy percentage (0 to 100%)
Display accuracy values at the end of each bar

Style: 
- Top bar (1D CNN) in bright gold/yellow (#FFD700) with "Best Model" legend
- All other bars in shades of green (#90EE90 to #32CD32)
- Add a red dashed vertical line marking the "Best Model" performance
- Clean, professional appearance suitable for academic publication
- Grid lines for easy reading
- Title "Model Performance Comparison" at the top in bold

Make bars thick and clearly visible with percentage labels at the end of each bar.
```

---

## 🔄 PROMPT 4: Methodology Flowchart

```
Create a vertical methodology flowchart for traffic flow prediction with the following steps:

1. START: Traffic Flow Prediction System (rounded box)
Arrow down to:

2. Data Collection: 5000 samples, 13 features (Traffic, Weather, Junction data)
Arrow down to:

3. Data Preprocessing: Handle missing values, Remove outliers, Label encoding
Arrow down to:

4. Feature Engineering: 19 features created (TimeOfDay, RushHour, Weather interactions)
Arrow down to:

5. Data Split: 80% Training (4000), 20% Testing (1000)
Arrow down to:

6. Model Training (bold box)
Split into 3 parallel branches:

Branch 1: Traditional ML (Random Forest, Decision Tree, SVM, LR, NB)
Branch 2 (CENTER - HIGHLIGHT): 1D CNN Proposed Model (4 Conv1D blocks, BatchNorm, Dropout, 200 epochs)
Branch 3: Transfer Learning (VGG16, VGG19, ResNet50)

All 3 branches converge down to:

7. Model Evaluation: Accuracy, Precision, Recall, F1, Confusion Matrix, Training Time
Arrow down to:

8. Statistical Testing: McNemar Test (p < 0.05)
Arrow down to:

9. Best Model Selection: 1D CNN: 92.80% (bold, highlighted)
Arrow down to:

10. Deployment: Streamlit Web Application (rounded box)

Use colors: Blue for start/end, light blue for data, yellow for preprocessing, orange for feature engineering, purple for split, green for models (bright green for 1D CNN - make it stand out), purple for evaluation, red for testing. All arrows flow downward with clear connections.
```

---

## 🎨 Post-Generation Adjustments

After AI generates each diagram:

### For All Diagrams:
1. **Highlight 1D CNN boxes** - Make them bright green with thick borders
2. **Bold important text** - "Model Training", "Best Model Selection"
3. **Adjust colors** if needed to match conference paper standards
4. **Align boxes** - Use Lucidchart's alignment tools
5. **Increase arrow thickness** - Make them 2-3pt wide
6. **Add title** above each diagram

### Color Codes (if manual adjustment needed):
- Light Blue: #AED6F1
- Yellow: #F9E79F
- Orange: #FAD7A0
- Light Green: #ABEBC6
- **Bright Green (1D CNN): #82E0AA** ← Make this stand out!
- Purple: #D7BDE2
- Light Red: #FADBD8
- Red: #F5B7B1
- Blue: #85C1E2

### Export Settings:
- Format: PNG
- Resolution: 300 DPI
- Size: Large (for paper quality)

---

## 🚀 Quick Tips

1. **If AI doesn't understand:** Break the prompt into smaller parts and generate step-by-step
2. **If layout is wrong:** Manually drag boxes to correct positions
3. **If colors are off:** Select boxes and change fill color manually
4. **If text is small:** Select all text → Increase to 11-12pt

---

## ✅ Checklist Before Exporting

- [ ] All boxes are present and correctly positioned
- [ ] 1D CNN boxes are highlighted (bright green, bold)
- [ ] Arrows show correct flow direction
- [ ] Text is readable (11pt minimum)
- [ ] Colors match the style
- [ ] Title is added
- [ ] Exported at 300 DPI

---

## 📝 Alternative: Manual Creation

If Lucidchart AI doesn't work well, refer back to:
`docs/LUCIDCHART_DIAGRAM_GUIDE.md`

That guide has step-by-step manual instructions for creating each diagram.

---

Good luck! The AI should do 80% of the work, then you just polish! 🎯
