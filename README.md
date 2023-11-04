# Waymo Motion

## Links

- [Waymo Motion Dataset GS](https://console.cloud.google.com/storage/browser/waymo_open_dataset_motion_v_1_2_0)

## Notes

- Use `gcloud compute ssh --project=high-perf-ml --zone=us-east4-a waymo-cpu-1` to connect to remote VM
- Use `gcloud compute scp --recurse . waymo-cpu-1:/home/steve/waymo` to push to remote VM.
- Make sure to have the `GOOGLE_APPLICATION_CREDENTIALS` in the GCP VM to access the Waymo dataset. (`export GOOGLE_APPLICATION_CREDENTIALS=credentials.json`)