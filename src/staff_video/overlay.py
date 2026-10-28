import cv2


COLORS = {"Tag observed": (115, 185, 10), "Staff tracked": (220, 145, 25), "Uncertain": (140, 140, 140)}


def annotate(frame, record):
    image = frame.copy()
    for person in record["people"]:
        x1, y1, x2, y2 = person["box"]
        color = COLORS[person["status"]]
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        identity = person["track_id"] or "unassigned"
        label = f'{person["status"]} #{identity}'
        cv2.putText(image, label, (x1, max(16, y1 - 7)), cv2.FONT_HERSHEY_SIMPLEX, .42, color, 1, cv2.LINE_AA)
        if person["staff_present"]:
            center = (round(person["center_x"]), round(person["center_y"]))
            cv2.circle(image, center, 4, color, -1)
            cv2.putText(image, f"xy {center[0]},{center[1]}", (x1, min(image.shape[0] - 8, y2 + 15)),
                        cv2.FONT_HERSHEY_SIMPLEX, .42, color, 1, cv2.LINE_AA)
        if person["tag_observed"] and person["tag_box"]:
            left, top, right, bottom = person["tag_box"]
            cv2.rectangle(image, (left, top), (right, bottom), (0, 220, 255), 2)
    text = f'Frame {record["frame_index"]} | {record["timestamp_seconds"]:.2f}s | staff {record["staff_count"]}'
    cv2.rectangle(image, (0, 0), (min(image.shape[1], 520), 30), (20, 30, 35), -1)
    cv2.putText(image, text, (10, 21), cv2.FONT_HERSHEY_SIMPLEX, .52, (255, 255, 255), 1, cv2.LINE_AA)
    return image
