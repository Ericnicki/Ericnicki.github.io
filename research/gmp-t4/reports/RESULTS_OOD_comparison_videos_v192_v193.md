# Six verified OOD videos: joint, swap and official GMP

2026-10-13. Delivered exactly six videos: one low-friction and one high-friction video per method. Each contains all twelve original evaluation trials, with visible physical resets, control clocks, native target outline and true terminal G labels. No training, ID task, new policy evaluation or outcome selection.

[Open all six videos](../../outputs/ood_comparison_videos_20261013/verified/ood_comparison_videos_v193_output/index.html) · [Local verification](../../outputs/ood_comparison_videos_20261013/v193_local_verification.json) · [Plan](../plans/RUN_PLAN_OOD_comparison_videos_v192.md).

The same boundary tasks are used across all methods: task0at low friction0.0025, task16at high friction0.0225. These deliberately show the two extremes of the requested50%friction OOD range and are not a representative frequency sample. Every displayed trial in these two extreme tasks failed for all three methods; the aggregate32task results remain in the preceding OOD reports.

| Method | Low friction0.0025 | High friction0.0225 |
| --- | --- | --- |
| Joint | [MP4, 161.0s](../../outputs/ood_comparison_videos_20261013/verified/ood_comparison_videos_v193_output/joint_low_friction_0.0025.mp4) | [MP4, 79.2s](../../outputs/ood_comparison_videos_20261013/verified/ood_comparison_videos_v193_output/joint_high_friction_0.0225.mp4) |
| Original swap | [MP4, 162.6s](../../outputs/ood_comparison_videos_20261013/verified/ood_comparison_videos_v193_output/swap_low_friction_0.0025.mp4) | [MP4, 78.2s](../../outputs/ood_comparison_videos_20261013/verified/ood_comparison_videos_v193_output/swap_high_friction_0.0225.mp4) |
| Official GMP + memory | [MP4, 122.5s](../../outputs/ood_comparison_videos_20261013/verified/ood_comparison_videos_v193_output/gmp_low_friction_0.0025.mp4) | [MP4, 76.1s](../../outputs/ood_comparison_videos_20261013/verified/ood_comparison_videos_v193_output/gmp_high_friction_0.0225.mp4) |

Videos are checked rendering replays of saved commands, rather than new runs of the policies. Joint/swap records come from accepted v186, GMP from accepted v187; all initial physics/friction snapshots match byte-exactly. Original memory adaptations remain represented in the saved commands; no memory is recomputed for visualization. The sidebar return list is a retrospective overlay, not information given to a policy.

Final rendering v193uses the native visualization camera pose with vertical field of view60degrees, keeping the cube visible during large overshoots. The projected cube center is inside the camera on every rendered frame. True native target bounds are overlaid in green. Each control frame corresponds to0.1physical seconds, encoded at10fps; one-second reset and terminal-result holds are added per trial. H264/yuv420p MP4,1024×640, with browser fast-start metadata. Native simulation/rendering/encoding runs only on c1d3019784aa through compute.sh.

All72final visualization trial replays /5356controls reproduce recorded TCP/cube poses and cube velocities, clocks, native judgments, requested/executed commands and terminal G. Maximum physical error0; all6796exported frames decode. Local archive/output/source hashes and all six gallery video/poster links pass; all six final decoded posters were visually inspected. Owned rendering processes49062and49419are absent.

The first completed renderer v192passed the same replay and decode checks but cropped large overshoots; its sources, outputs and locally verified archive are preserved. v193rerenders the same72trial histories with a wider view only. Total144visualization trial replays across the two versions, zero new policy evaluations or training. Existing research snapshots and outcome statistics remain unchanged.

Final [archive](../../outputs/ood_comparison_videos_20261013/v193.tar.gz) contains33files including the manifest, with source and parent-data bindings. SHA256 `6af0a7e9e88f5e388948493cbf339bcec50a098e80b4c98756a884d8d394f857`. Parent raw records remain in their previously delivered research archives.
