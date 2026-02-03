# Lucidchart Diagram Guide for Conference Paper

## 🎨 Tools Needed
- **Lucidchart**: https://lucid.app/
- Create a free account
- Use templates: Flowchart, System Architecture, PRISMA

---

## 📊 DIAGRAM 1: Systematic Literature Review (PRISMA Flowchart)

### Layout: Vertical Flow (Top to Bottom)

### Boxes to Create:

#### **Box 1** (Top - Blue)
```
Identification
Database Search
(IEEE, Springer, ScienceDirect, ACM)
n = 450 papers
```
**Style**: Rectangle, Light Blue (#E8F4F8), Border: Black

↓ (Arrow down)

#### **Box 2** (Light Blue)
```
Screening
Title & Abstract Review
n = 450
```
**Style**: Rectangle, Light Blue (#D4E6F1), Border: Black

→ (Arrow right to exclusion box)

#### **Box 3** (Right side - Red)
```
Excluded (n=280)
• Not traffic prediction
• Different domain
```
**Style**: Rectangle, Light Red (#FADBD8), Border: Black

↓ (Arrow down from Box 2)

#### **Box 4** (Light Blue)
```
Eligibility
Full-Text Assessment
n = 170
```
**Style**: Rectangle, Light Blue (#D4E6F1), Border: Black

→ (Arrow right to exclusion box)

#### **Box 5** (Right side - Red)
```
Excluded (n=120)
• No ML/DL methods
• Insufficient data
• Poor methodology
```
**Style**: Rectangle, Light Red (#FADBD8), Border: Black

↓ (Arrow down from Box 4)

#### **Box 6** (Green)
```
Included
Final Papers for Review
n = 50
```
**Style**: Rectangle, Green (#A9DFBF), Border: Black

↓ (Arrow down)

#### **Box 7** (Bold - Blue)
```
Analysis Categories
```
**Style**: Rectangle, Blue (#85C1E2), Border: Black (thick), **Bold Text**

↓ (Arrows down to 3 boxes)

#### **Box 8, 9, 10** (Three boxes side-by-side - Light Green)
```
Box 8:
Traditional ML
(n=15)
RF, SVM, DT

Box 9:
Deep Learning
(n=25)
CNN, LSTM, RNN

Box 10:
Hybrid Models
(n=10)
Combined approaches
```
**Style**: Rectangle, Light Green (#ABEBC6), Border: Black

### Title Above Diagram:
**"Systematic Literature Review Process (PRISMA Framework)"**

---

## 🏗️ DIAGRAM 2: System Architecture

### Layout: Layered Architecture (Top to Bottom)

### Layer 1: Data Collection (Top)
Create 3 boxes horizontally:

```
Box 1: Traffic Data Collection
Box 2: Weather Data
Box 3: Junction Info
```
**Style**: Rectangle, Light Blue (#AED6F1), Border: Black

↓ (Arrows from all 3 converging down)

### Layer 2: Preprocessing
```
Data Preprocessing
• Missing value handling
• Outlier removal
• Normalization
```
**Style**: Rectangle, Yellow (#F9E79F), Border: Black

↓ (Arrow down)

### Layer 3: Feature Engineering
```
Feature Engineering
• Temporal features (Hour, Day)
• Interaction features
• Categorical encoding
```
**Style**: Rectangle, Orange (#FAD7A0), Border: Black

↓ (Arrow down, splits into 3)

### Layer 4: Model Layer (3 boxes horizontally)
```
Box 1:
Traditional ML
• Random Forest
• Decision Tree
• SVM

Box 2 (CENTER - HIGHLIGHT):
1D CNN
(Proposed)
• Conv1D layers
• Batch Norm
• Dropout

Box 3:
Transfer Learning
• VGG16
• VGG19
• ResNet50
```
**Style**: 
- Box 1 & 3: Light Green (#ABEBC6), Border: Black
- **Box 2: Bright Green (#82E0AA), Border: Black (thick), Bold Text** ← HIGHLIGHT THIS

↓ (Arrows from all 3 converging down)

### Layer 5: Evaluation
```
Model Evaluation
• Accuracy: 92.80%
• Precision, Recall, F1
• McNemar Test
```
**Style**: Rectangle, Purple (#D7BDE2), Border: Black

↓ (Arrow down)

### Layer 6: Output (Bottom)
```
Traffic Prediction
Low | Medium | High | Severe
```
**Style**: Rectangle, Red (#F5B7B1), Border: Black

### Title Above Diagram:
**"Proposed System Architecture for Traffic Flow Prediction"**

---

## 🔄 DIAGRAM 3: Methodology Flowchart

### Layout: Vertical Flow with Branching

### Boxes to Create:

#### **Box 1** (Top)
```
START
Traffic Flow Prediction System
```
**Style**: Rounded Rectangle, Blue (#85C1E2), Border: Black

↓

#### **Box 2**
```
Data Collection
5000 samples, 13 features
Traffic, Weather, Junction data
```
**Style**: Rectangle, Light Blue (#AED6F1)

↓

#### **Box 3**
```
Data Preprocessing
• Handle missing values
• Remove outliers
• Label encoding
```
**Style**: Rectangle, Yellow (#F9E79F)

↓

#### **Box 4**
```
Feature Engineering
19 features created
• TimeOfDay, RushHour
• Weather interactions
```
**Style**: Rectangle, Orange (#FAD7A0)

↓

#### **Box 5**
```
Data Split
80% Training (4000)
20% Testing (1000)
```
**Style**: Rectangle, Purple (#D7BDE2)

↓

#### **Box 6** (Bold)
```
Model Training
```
**Style**: Rectangle, Blue (#85C1E2), **Bold**, Thick Border

↓ (Splits into 3 branches)

#### **Three Parallel Boxes** (Side by side)
```
Left:
Traditional ML
• Random Forest
• Decision Tree
• SVM, LR, NB

CENTER (HIGHLIGHT):
1D CNN
(Proposed Model)
• 4 Conv1D blocks
• BatchNorm
• Dropout
• 200 epochs

Right:
Transfer Learning
• VGG16
• VGG19
• ResNet50
```
**Style**: 
- Left/Right: Light Green (#ABEBC6)
- **CENTER: Bright Green (#82E0AA), Bold, Thick Border** ← HIGHLIGHT

↓ (All 3 converge)

#### **Box 7**
```
Model Evaluation
• Accuracy, Precision, Recall, F1
• Confusion Matrix
• Training Time
```
**Style**: Rectangle, Purple (#D7BDE2)

↓

#### **Box 8**
```
Statistical Testing
McNemar Test
(p < 0.05)
```
**Style**: Rectangle, Red (#F5B7B1)

↓

#### **Box 9** (Bold - Highlight)
```
Best Model Selection
1D CNN: 92.80%
```
**Style**: Rectangle, Bright Green (#82E0AA), **Bold**, Thick Border

↓

#### **Box 10** (End)
```
Deployment
Streamlit Web Application
```
**Style**: Rounded Rectangle, Blue (#85C1E2)

### Title Above Diagram:
**"Complete Methodology Flowchart"**

---

## 🎨 Lucidchart Tips

### Getting Started:
1. Go to https://lucid.app/
2. Click "New" → "Lucidchart"
3. Choose "Blank Document"

### Creating Boxes:
- Press **"R"** for rectangle
- Press **"Shift + R"** for rounded rectangle
- Drag to create

### Styling Boxes:
1. Click box
2. Right panel → **Fill Color** (use hex codes above)
3. Border → **Line Width**: 2pt
4. For highlighted boxes: **Line Width**: 4pt

### Adding Text:
- Double-click box to type
- Select text → **Bold** for emphasis
- Font size: 11-12pt for normal, 14pt for titles

### Creating Arrows:
- Press **"L"** for line
- Select line → **Arrow End**: Arrow
- Style → **Line Width**: 2-3pt

### Alignment:
- Select multiple boxes
- Top menu → **Align** → Center/Distribute

### Exporting:
1. File → **Download**
2. Format: **PNG** or **PDF**
3. Resolution: **300 DPI** (for paper)
4. Size: **Large** or **Custom**

---

## 📏 Recommended Sizes

### For Conference Paper (IEEE/Springer):
- **Width**: 7.5 inches (single column) or 16 inches (double column)
- **Height**: As needed (usually 10-14 inches)
- **Resolution**: 300 DPI minimum
- **Format**: PNG or PDF

### Box Sizes:
- Standard box: 150-200 pixels wide
- Highlighted boxes: Add 10px padding
- Font: 11-12pt for text, 14pt for titles

---

## ✅ Quality Checklist

Before exporting, verify:
- [ ] All text is readable (11pt minimum)
- [ ] Arrows clearly show flow direction
- [ ] Highlighted boxes (1D CNN) stand out
- [ ] Colors match the guide
- [ ] No spelling errors
- [ ] Title is present and bold
- [ ] All boxes are aligned
- [ ] Exported at 300 DPI

---

## 🎯 Quick Start Order

1. **Start with Diagram 3** (Methodology) - Easiest
2. **Then Diagram 2** (Architecture) - Medium difficulty
3. **Finally Diagram 1** (SLR) - Most boxes

Estimated time: 2-3 hours total for all 3 diagrams

Good luck! 🚀
