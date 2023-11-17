# Waymo Motion

## Training

### v0.1.0

Modalities
- [x] Agent data (T_all x 128 x 8)
- [ ] Traffic Light State (T_obs x 16 x 6)
- [ ] Road Graph (30000, 5)
- [ ] LiDAR

Architecture
- Regular Attention

Training
- Loss is MSE after static latent dimension reducer layer
- Weighted average of future predictions
- No masking

Results
- Model is not really learning anything through ~600 examples.
- Each mini-batch takes a few seconds to run.
- Can handle mini-batch size of 6 on a T4 GPU.

Recent Updates
- 1. Include past states into decoder (81 => 91)
- 2. Add other modalities (road graph)
- 3. Introduce Performer to speed up training.
    - need to get masked training to work

Possible Directions
- 1. Change the way that the road graph is used
    - don't flatten and concat
    - give its own encoder to be used for cross-attention with agents
- 2. Add other modality (traffic light)
- 3. Switch to trajectory-based loss function
- 4. Use number of agents as sequence and time as hidden dim

## Links

- [Waymo Motion Dataset GS](https://console.cloud.google.com/storage/browser/waymo_open_dataset_motion_v_1_2_0)

## Development Notes

- Use `gcloud compute ssh --project=high-perf-ml --zone=us-east4-a waymo-cpu-1` to connect to remote VM
- Use `gcloud compute scp --recurse . waymo-cpu-1:/home/steve/waymo` to push to remote VM.
- Make sure to have the `GOOGLE_APPLICATION_CREDENTIALS` in the GCP VM to access the Waymo dataset. (`export GOOGLE_APPLICATION_CREDENTIALS=credentials.json`)