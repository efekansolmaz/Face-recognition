import numpy as np

class TrackedFace:
    def __init__(self, track_id: int, bbox: list, face_raw: np.ndarray):
        self.track_id = track_id
        self.bbox = bbox  # [x, y, w, h]
        self.face_raw = face_raw  # 15 elemanli YuNet ciktisi
        self.worker_id = None
        self.worker_code = None
        self.worker_name = None
        self.worker_department = "Genel"
        self.emotion = "Nötr"
        self.emotion_conf = 0.0
        self.emotion_color = (200, 200, 200)
        self.last_log_time = 0.0
        self.missed_frames = 0

class FaceTracker:
    def __init__(self, max_missed: int = 12, iou_threshold: float = 0.30):
        self.tracks: dict[int, TrackedFace] = {}
        self.next_track_id = 1
        self.max_missed = max_missed
        self.iou_threshold = iou_threshold

    @staticmethod
    def _compute_iou(boxA, boxB):
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
        yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])

        interW = max(0, xB - xA)
        interH = max(0, yB - yA)
        interArea = interW * interH

        boxAArea = boxA[2] * boxA[3]
        boxBArea = boxB[2] * boxB[3]
        unionArea = boxAArea + boxBArea - interArea

        if unionArea == 0:
            return 0.0
        return interArea / unionArea

    def update(self, detected_faces: list[np.ndarray]) -> list[TrackedFace]:
        updated_tracks = []
        unmatched_detections = []

        if not detected_faces:
            for t_id in list(self.tracks.keys()):
                self.tracks[t_id].missed_frames += 1
                if self.tracks[t_id].missed_frames > self.max_missed:
                    del self.tracks[t_id]
            return []

        track_ids = list(self.tracks.keys())
        used_tracks = set()

        for face in detected_faces:
            bbox = [int(face[0]), int(face[1]), int(face[2]), int(face[3])]
            best_iou = 0.0
            best_track_id = None

            for t_id in track_ids:
                if t_id in used_tracks:
                    continue
                iou = self._compute_iou(bbox, self.tracks[t_id].bbox)
                if iou > best_iou:
                    best_iou = iou
                    best_track_id = t_id

            if best_track_id is not None and best_iou >= self.iou_threshold:
                track = self.tracks[best_track_id]
                track.bbox = bbox
                track.face_raw = face
                track.missed_frames = 0
                used_tracks.add(best_track_id)
                updated_tracks.append(track)
            else:
                unmatched_detections.append(face)

        for t_id in track_ids:
            if t_id not in used_tracks:
                self.tracks[t_id].missed_frames += 1
                if self.tracks[t_id].missed_frames > self.max_missed:
                    del self.tracks[t_id]

        for face in unmatched_detections:
            bbox = [int(face[0]), int(face[1]), int(face[2]), int(face[3])]
            new_track = TrackedFace(self.next_track_id, bbox, face)
            self.tracks[self.next_track_id] = new_track
            self.next_track_id += 1
            updated_tracks.append(new_track)

        return updated_tracks
