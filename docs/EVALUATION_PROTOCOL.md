# UAV-VisLoc Evaluation Protocol

## Task: Absolute Visual Localization (AVL)

**UAV-VisLoc is NOT VPR - it's AVL!**

- **Input**: Drone image
- **Output**: Predicted GPS coordinates (latitude, longitude)
- **Evaluation**: Compare predicted coordinates to ground truth GPS coordinates

## Metrics

### R@1 (Recall@1)

- Percentage of queries where predicted coordinates are within a threshold distance from ground truth
- Common thresholds: 5m, 10m, 25m
- Formula: `R@1 = (predictions within threshold) / (total queries) * 100%`

### Dis@1 (Distance@1)

- Average localization error in meters for all queries
- Computed as: `Dis@1 = mean(haversine_distance(pred_lat, pred_lon, gt_lat, gt_lon))`

## Evaluation Protocol

1. For each drone image (query):

   - Method predicts GPS coordinates (lat, lon)
   - Compare with ground truth GPS coordinates from CSV
   - Compute distance error using Haversine formula

2. Aggregate metrics:
   - **R@1**: Percentage within threshold (default: 5m)
   - **Dis@1**: Mean distance error in meters

## Difference from VPR

| Aspect         | VPR (Visual Place Recognition) | AVL (Absolute Visual Localization)     |
| -------------- | ------------------------------ | -------------------------------------- |
| **Output**     | Closest reference image index  | GPS coordinates (lat, lon)             |
| **Evaluation** | Retrieval accuracy (R@K)       | Localization error (meters)            |
| **Reference**  | Database of reference images   | Satellite maps or coordinate space     |
| **Metrics**    | R@1, R@5, R@10                 | R@1 (within threshold), Dis@1 (meters) |

## Implementation Strategy

**Important**: Methods only need to predict coordinates - VPR backbone is optional!

Methods can use different approaches:

1. **VPR-based approach** (optional):

   - Extract patches from satellite map at candidate locations
   - Match drone image to patches using VPR
   - Select best match location → predict coordinates

2. **Direct coordinate prediction** (simpler):

   - Predict coordinates directly from drone image
   - Can use satellite map for refinement/verification
   - No reference database needed

3. **Hybrid (like FoundLoc)**:
   - VPR retrieves candidate locations (optional)
   - Fine-grained matching refines to exact coordinates
   - Output: GPS coordinates

**AnyLoc Note**: AnyLoc has NOT been evaluated on UAV-VisLoc in published papers. AnyLoc is a VPR method (retrieves closest matches), so it would need adaptation to predict coordinates. AnyLoc-GeM baseline will need to:

- Use VPR to match drone image → satellite patches
- Select best match location
- Output coordinates (lat, lon) of matched patch

## Current SOTA

- **ViT-Base/16 (trained)**: R@1=84.95%, Dis@1=149.07m (same-area)
- Our goal: Beat this with a **training-free** method
