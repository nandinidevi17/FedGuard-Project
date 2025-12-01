# """
# insecure_server.py (final patched)

# Features:
# - Robust deserialization: JSON, relaxed JSON, ast.literal_eval, pickle,
#   numpy npy/npz (direct and base64), gzip/zlib decompression.
# - Shape validation against the loaded Keras model; rejects mismatched payloads.
# - Skips duplicate messages (partition+offset).
# - Per-client invalid-count tracking and temporary bans to prevent log flood.
# - Insecure aggregation semantics (accepts all valid clients) but validates payloads to avoid crashes.
# """

# import os
# import sys
# import json
# import pickle
# import io
# import ast
# import base64
# import re
# import gzip
# import zlib
# import time
# import numpy as np
# import tensorflow as tf
# from kafka import KafkaConsumer
# import logging
# from datetime import datetime, timedelta

# # Add project root to path to import config
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# from config import *  # expects KAFKA_SERVER, KAFKA_TOPIC, MODELS_DIR, LOGS_DIR, NUM_CLIENTS, NUM_FEDERATED_ROUNDS, LOG_LEVEL, LOG_FORMAT

# os.makedirs(LOGS_DIR, exist_ok=True)

# logging.basicConfig(
#     level=getattr(logging, LOG_LEVEL),
#     format=LOG_FORMAT,
#     handlers=[
#         logging.FileHandler(os.path.join(LOGS_DIR, 'insecure_server.log'), encoding='utf-8'),
#         logging.StreamHandler(sys.stdout)
#     ]
# )
# logger = logging.getLogger(__name__)

# logger.warning("="*60)
# logger.warning("[WARNING] INSECURE SERVER (NO DEFENSE)")
# logger.warning("="*60)
# logger.warning("This server accepts client updates (but will validate payload shapes to avoid crashes).")
# logger.warning(f"Expected clients: {NUM_CLIENTS}")


# # --------------------------
# # Utility deserialization helpers
# # --------------------------
# def try_json_bytes(b):
#     try:
#         return json.loads(b.decode('utf-8'))
#     except Exception:
#         return None

# def try_json_string(s):
#     try:
#         return json.loads(s)
#     except Exception:
#         return None

# def try_relaxed_json_from_string(s):
#     # If single quotes dominate, try swapping them for double quotes (naive but often works for repr-like JSON)
#     if not isinstance(s, str):
#         return None
#     try:
#         if s.count("'") > s.count('"'):
#             candidate = s.replace("'", '"')
#             return json.loads(candidate)
#     except Exception:
#         pass
#     return None

# def try_ast_literal(s):
#     if not isinstance(s, str):
#         return None
#     try:
#         return ast.literal_eval(s)
#     except Exception:
#         return None

# def try_pickle_bytes(b):
#     try:
#         return pickle.loads(b)
#     except Exception:
#         return None

# def try_numpy_load_bytes(b):
#     if not isinstance(b, (bytes, bytearray)):
#         return None
#     bio = io.BytesIO(b)
#     try:
#         bio.seek(0)
#         arr = np.load(bio, allow_pickle=True)
#         if isinstance(arr, np.ndarray):
#             return arr.tolist()
#         out = {}
#         for kk in arr.files:
#             out[kk] = arr[kk].tolist()
#         return out
#     except Exception:
#         return None

# _b64_re = re.compile(r'^[A-Za-z0-9+/=\s]+$')
# def looks_like_base64_string(s):
#     if not isinstance(s, str):
#         return False
#     if len(s) < 100:
#         return False
#     ss = ''.join(s.split())
#     return bool(_b64_re.fullmatch(ss))

# def try_base64_then_numpy_or_json(s):
#     if not isinstance(s, str):
#         return None
#     ss = ''.join(s.split())
#     try:
#         raw = base64.b64decode(ss)
#     except Exception:
#         return None
#     # try numpy load first
#     n = try_numpy_load_bytes(raw)
#     if n is not None:
#         return n
#     # then try JSON
#     try:
#         return json.loads(raw.decode('utf-8', errors='replace'))
#     except Exception:
#         return None

# def try_gzip_bytes(b):
#     try:
#         return gzip.decompress(b)
#     except Exception:
#         return None

# def try_zlib_bytes(b):
#     try:
#         return zlib.decompress(b)
#     except Exception:
#         return None

# # Master safe deserialize
# def safe_deserialize(raw):
#     """
#     Try many strategies to turn raw Kafka message.value into a usable Python object.
#     Returns Python object or None.
#     """
#     if raw is None:
#         return None

#     # If it's already a Python object
#     if isinstance(raw, (dict, list, tuple, str, int, float, np.ndarray)):
#         return raw

#     # If raw is bytes/bytearray -> try ordered strategies
#     if isinstance(raw, (bytes, bytearray)):
#         # 1) strict JSON bytes
#         j = try_json_bytes(raw)
#         if j is not None:
#             return j
#         # 2) try pickle
#         p = try_pickle_bytes(raw)
#         if p is not None:
#             return p
#         # 3) numpy binary (npy/npz)
#         n = try_numpy_load_bytes(raw)
#         if n is not None:
#             return n
#         # 4) try gzip then numpy/json/pickle
#         g = try_gzip_bytes(raw)
#         if g is not None:
#             inner = safe_deserialize(g)
#             if inner is not None:
#                 return inner
#         # 5) try zlib
#         z = try_zlib_bytes(raw)
#         if z is not None:
#             inner = safe_deserialize(z)
#             if inner is not None:
#                 return inner
#         # 6) decode to string and try string-based strategies
#         try:
#             s = raw.decode('utf-8', errors='replace')
#         except Exception:
#             s = None
#         if s:
#             # a) strict JSON
#             j2 = try_json_string(s)
#             if j2 is not None:
#                 return j2
#             # b) relaxed JSON (single-quote)
#             rj = try_relaxed_json_from_string(s)
#             if rj is not None:
#                 return rj
#             # c) ast.literal_eval
#             a = try_ast_literal(s)
#             if a is not None:
#                 return a
#             # d) base64-wrapped numpy/json
#             if looks_like_base64_string(s):
#                 bnp = try_base64_then_numpy_or_json(s)
#                 if bnp is not None:
#                     return bnp
#             # fallback: return the raw string (so we can preview)
#             return s

#     # fallback: string representation
#     try:
#         return str(raw)
#     except Exception:
#         return None


# # --------------------------
# # Normalization to list[np.array]
# # --------------------------
# def normalize_weights_payload(payload):
#     """
#     Convert payload into list of numpy arrays (one per model weight tensor).
#     Accepts:
#       - list/tuple of nested lists or numeric arrays
#       - dict with 'weights' or arr_*
#       - single numpy ndarray (converted to single-layer list)
#     Returns: list[np.array] or None
#     """
#     if payload is None:
#         return None

#     # If it's a string that is still JSON-like, attempt parse
#     if isinstance(payload, str):
#         parsed = None
#         try:
#             parsed = json.loads(payload)
#         except Exception:
#             try:
#                 parsed = ast.literal_eval(payload)
#             except Exception:
#                 parsed = None
#         if parsed is not None:
#             payload = parsed

#     # If dict with weights key
#     if isinstance(payload, dict):
#         for key in ('weights', 'model_weights', 'params', 'w', 'payload', 'data'):
#             if key in payload:
#                 payload = payload[key]
#                 break

#     # If it's numpy array -> single layer
#     if isinstance(payload, np.ndarray):
#         try:
#             return [np.array(payload, dtype=np.float32)]
#         except Exception:
#             return None

#     # If dict with arr_ keys (npz-like), convert ordered
#     if isinstance(payload, dict) and any(k.startswith('arr_') for k in payload.keys()):
#         try:
#             ordered_keys = sorted(payload.keys())
#             return [np.array(payload[k], dtype=np.float32) for k in ordered_keys]
#         except Exception:
#             return None

#     # If list/tuple
#     if isinstance(payload, (list, tuple)):
#         try:
#             out = []
#             for layer in payload:
#                 if isinstance(layer, (list, tuple, np.ndarray)):
#                     arr = np.array(layer, dtype=np.float32)
#                 elif isinstance(layer, dict) and any(k.startswith('arr_') for k in layer.keys()):
#                     # nested dict representing a single layer? try to flatten single arr entry
#                     ordered_keys = sorted(layer.keys())
#                     if len(ordered_keys) == 1:
#                         arr = np.array(layer[ordered_keys[0]], dtype=np.float32)
#                     else:
#                         return None
#                 else:
#                     # unsupported element type
#                     return None
#                 out.append(arr)
#             return out
#         except Exception:
#             return None

#     return None


# # --------------------------
# # Main server
# # --------------------------
# def main():
#     # Load model
#     try:
#         model_path = os.path.join(MODELS_DIR, "ucsd_baseline.h5")
#         logger.info(f"\nLoading model from {model_path}...")
#         global_model = tf.keras.models.load_model(model_path)
#         logger.info("[OK] Model loaded")
#     except Exception:
#         logger.exception("Failed to load base model - aborting")
#         return

#     expected_weights = global_model.get_weights()
#     expected_num_layers = len(expected_weights)
#     expected_shapes = [w.shape for w in expected_weights]
#     logger.info(f"Model expects {expected_num_layers} weight arrays with shapes: {expected_shapes}")

#     # Initialize Kafka consumer
#     try:
#         consumer = KafkaConsumer(
#             KAFKA_TOPIC,
#             bootstrap_servers=KAFKA_SERVER,
#             value_deserializer=lambda m: m,  # receive raw bytes/object
#             auto_offset_reset='latest',
#             enable_auto_commit=True,
#             group_id='insecure-server-group'
#         )
#         logger.info(f"\nConnecting to Kafka at {KAFKA_SERVER} (topic={KAFKA_TOPIC})... [auto_offset_reset=latest]")
#         logger.info("[OK] Kafka connected")
#     except Exception:
#         logger.exception("Failed to create Kafka consumer - aborting")
#         return

#     logger.info("\n" + "="*60)
#     logger.info("SERVER RUNNING (INSECURE - accepting updates but validating payloads)")
#     logger.info("="*60 + "\n")

#     client_updates = {}
#     round_number = 0

#     # Duplicate detection and client throttling
#     processed_offsets = set()  # (partition, offset)
#     invalid_counts = {}        # client_id -> count
#     ban_list = {}              # client_id -> ban_until_timestamp
#     INVALID_THRESHOLD = 10     # after this many invalid messages, temporarily ban client
#     BAN_SECONDS = 300          # ban duration (5 minutes)

#     try:
#         for message in consumer:
#             partition = getattr(message, 'partition', None)
#             offset = getattr(message, 'offset', None)
#             msg_key = message.key
#             raw_value = message.value

#             # Skip duplicate
#             po = (partition, offset)
#             if po in processed_offsets:
#                 logger.debug(f"Skipping duplicate message partition={partition} offset={offset}")
#                 continue
#             processed_offsets.add(po)

#             # Resolve key/client label
#             try:
#                 if isinstance(msg_key, (bytes, bytearray)):
#                     key_str = msg_key.decode('utf-8', errors='ignore')
#                 else:
#                     key_str = str(msg_key)
#             except Exception:
#                 key_str = f"unknown-key-{partition}-{offset}"

#             # Quick ban check
#             now_ts = time.time()
#             if key_str in ban_list and now_ts < ban_list[key_str]:
#                 logger.info(f"Skipping message from banned client {key_str} until {datetime.fromtimestamp(ban_list[key_str])}")
#                 continue
#             elif key_str in ban_list and now_ts >= ban_list[key_str]:
#                 del ban_list[key_str]
#                 invalid_counts.pop(key_str, None)
#                 logger.info(f"Ban lifted for client {key_str}")

#             # Attempt robust deserialization
#             update_data = safe_deserialize(raw_value)
#             if update_data is None:
#                 # Log a preview for operator - raw bytes might be binary
#                 preview = None
#                 try:
#                     if isinstance(raw_value, (bytes, bytearray)):
#                         # show first 200 bytes hex or text attempt
#                         try_text = raw_value[:200].decode('utf-8', errors='replace')
#                         preview = try_text
#                     else:
#                         preview = str(raw_value)[:200]
#                 except Exception:
#                     preview = "<unprintable>"
#                 logger.info(f"\n[Round {round_number + 1}] Received update (but failed to deserialize) key={key_str} (partition={partition} offset={offset})")
#                 logger.info(f"  Payload preview (first 200 chars): {preview}")
#                 # increase invalid count for key_str
#                 invalid_counts[key_str] = invalid_counts.get(key_str, 0) + 1
#                 if invalid_counts[key_str] >= INVALID_THRESHOLD:
#                     ban_list[key_str] = time.time() + BAN_SECONDS
#                     logger.warning(f"Client {key_str} exceeded invalid threshold; temporarily banning until {datetime.fromtimestamp(ban_list[key_str])}")
#                 continue

#             # Show short preview if string payload
#             if isinstance(update_data, str):
#                 preview = update_data[:200].replace('\n', '\\n')
#                 logger.info(f"\n[Round {round_number + 1}] Received update from key: {key_str} (partition={partition} offset={offset})")
#                 logger.info(f"  Metadata preview: <class 'str'>; payload preview (first 200 chars): {preview}")
#             else:
#                 logger.info(f"\n[Round {round_number + 1}] Received update from key: {key_str} (partition={partition} offset={offset})")
#                 try:
#                     if isinstance(update_data, dict):
#                         preview_meta = {k: update_data[k] for k in list(update_data.keys())[:6] if k != 'weights'}
#                     else:
#                         preview_meta = f"{type(update_data).__name__}"
#                 except Exception:
#                     preview_meta = "unavailable"
#                 logger.info(f"  Metadata preview: {preview_meta}")

#             # Determine client_id (prefer explicit field, else use key)
#             client_id = None
#             if isinstance(update_data, dict):
#                 client_id = update_data.get('client_id') or update_data.get('client') or update_data.get('sender') or update_data.get('id')
#             if not client_id:
#                 client_id = key_str or f"unknown-{partition}-{offset}"

#             # Extract weights payload candidate
#             weights_payload = None
#             if isinstance(update_data, dict):
#                 for k in ('weights', 'model_weights', 'params', 'w', 'payload', 'data'):
#                     if k in update_data:
#                         weights_payload = update_data[k]
#                         break
#                 if weights_payload is None and any(k.startswith('arr_') for k in update_data.keys()):
#                     weights_payload = update_data
#                 if weights_payload is None and isinstance(update_data.get('data'), (list, dict)):
#                     weights_payload = update_data.get('data')
#             elif isinstance(update_data, (list, tuple, np.ndarray)):
#                 weights_payload = update_data
#             else:
#                 weights_payload = update_data

#             normalized = normalize_weights_payload(weights_payload)
#             layer_count = 0 if normalized is None else len(normalized)
#             logger.info(f"  Received weights (layers: {layer_count}) from {client_id}")

#             if normalized is None:
#                 # invalid weight payload -> increment invalid counter for that client_id
#                 invalid_counts[client_id] = invalid_counts.get(client_id, 0) + 1
#                 logger.warning(f"  Invalid weight payload from {client_id} (invalid count={invalid_counts[client_id]})")
#                 if invalid_counts[client_id] >= INVALID_THRESHOLD:
#                     ban_list[client_id] = time.time() + BAN_SECONDS
#                     logger.warning(f"Client {client_id} exceeded invalid threshold; temporarily banning until {datetime.fromtimestamp(ban_list[client_id])}")
#                 continue

#             # store the normalized arrays for aggregation
#             client_updates[client_id] = normalized
#             with_weights_count = sum(1 for v in client_updates.values() if isinstance(v, list) and v)
#             logger.info(f"  Updates collected (with weights): {with_weights_count}/{NUM_CLIENTS}")

#             # Aggregate when we have enough valid client updates
#             if with_weights_count >= NUM_CLIENTS:
#                 round_number += 1
#                 logger.warning(f"\n[ROUND {round_number}] INSECURE AGGREGATION - accepting all clients (but validating shapes)")

#                 # Validate shapes and collect only valid items
#                 valid_items = {}
#                 for cid, wlist in client_updates.items():
#                     if not isinstance(wlist, list):
#                         logger.warning(f"Skipping {cid}: payload not a list of layers")
#                         continue
#                     if len(wlist) != expected_num_layers:
#                         logger.warning(f"Skipping {cid}: wrong number of layers ({len(wlist)}) expected {expected_num_layers}")
#                         continue
#                     bad_shape = False
#                     for idx, layer in enumerate(wlist):
#                         got_shape = getattr(layer, 'shape', None)
#                         if got_shape != expected_shapes[idx]:
#                             logger.warning(f"Skipping {cid}: shape mismatch at layer {idx} expected {expected_shapes[idx]} got {got_shape}")
#                             bad_shape = True
#                             break
#                     if bad_shape:
#                         continue
#                     try:
#                         np_weights = [np.array(layer, dtype=np.float32) for layer in wlist]
#                     except Exception:
#                         logger.warning(f"Skipping {cid}: failed to convert layers to numpy arrays")
#                         continue
#                     valid_items[cid] = np_weights

#                 if not valid_items:
#                     logger.error("No valid client payloads to aggregate this round; clearing stored updates and continuing.")
#                     client_updates = {}
#                     continue

#                 logger.info(f"  Averaging updates from: {', '.join(valid_items.keys())}")

#                 try:
#                     averaged_weights = []
#                     for layer_idx in range(expected_num_layers):
#                         stack = np.stack([valid_items[cid][layer_idx] for cid in valid_items.keys()], axis=0)
#                         averaged_layer = np.mean(stack, axis=0)
#                         averaged_weights.append(averaged_layer)
#                 except Exception:
#                     logger.exception("Error while averaging weights; skipping aggregation for this round")
#                     client_updates = {}
#                     continue

#                 try:
#                     global_model.set_weights(averaged_weights)
#                     save_path = os.path.join(MODELS_DIR, 'global_model_insecured.h5')
#                     global_model.save(save_path)
#                     logger.info(f"[OK] Model updated and saved to: {save_path}\n")
#                 except Exception:
#                     logger.exception("Failed to apply averaged weights to model - skipping save for this round")

#                 # reset collected updates for next round
#                 client_updates = {}
#                 # reset per-client invalid counters for clients that successfully contributed
#                 for cid in valid_items.keys():
#                     invalid_counts.pop(cid, None)
#                     ban_list.pop(cid, None)

#                 if round_number >= NUM_FEDERATED_ROUNDS:
#                     logger.info(f"Completed {NUM_FEDERATED_ROUNDS} rounds - stopping")
#                     break

#     except KeyboardInterrupt:
#         logger.info("\nKeyboard interrupt - stopping insecure server")
#     except Exception:
#         logger.exception("FATAL ERROR (insecure server)")
#     finally:
#         try:
#             consumer.close()
#         except Exception:
#             pass
#         logger.warning(f"\nInsecure server stopped after {round_number} rounds")


# if __name__ == "__main__":
#     main()

"""
Insecure Federated Learning Server (Fixed - Compatible with evaluation).

CRITICAL FIXES:
1. Loads the SAME baseline model architecture as secure server
2. Saves in compatible .h5 format
3. Uses proper model.save() instead of save_weights()
"""

import os
import sys
import json
import pickle
import io
import ast
import base64
import re
import gzip
import zlib
import time
import numpy as np
import tensorflow as tf
from kafka import KafkaConsumer
import logging
from datetime import datetime, timedelta

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import *

os.makedirs(LOGS_DIR, exist_ok=True)

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format=LOG_FORMAT,
    handlers=[
        logging.FileHandler(os.path.join(LOGS_DIR, 'insecure_server.log'), encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

logger.warning("="*60)
logger.warning("[WARNING] INSECURE SERVER (NO DEFENSE)")
logger.warning("="*60)
logger.warning("This server accepts ALL client updates without security checks.")
logger.warning(f"Expected clients: {NUM_CLIENTS}")


# ========================================
# DESERIALIZATION UTILITIES (Same as before)
# ========================================
def try_json_bytes(b):
    try:
        return json.loads(b.decode('utf-8'))
    except Exception:
        return None

def try_json_string(s):
    try:
        return json.loads(s)
    except Exception:
        return None

def try_relaxed_json_from_string(s):
    if not isinstance(s, str):
        return None
    try:
        if s.count("'") > s.count('"'):
            candidate = s.replace("'", '"')
            return json.loads(candidate)
    except Exception:
        pass
    return None

def try_ast_literal(s):
    if not isinstance(s, str):
        return None
    try:
        return ast.literal_eval(s)
    except Exception:
        return None

def try_pickle_bytes(b):
    try:
        return pickle.loads(b)
    except Exception:
        return None

def try_numpy_load_bytes(b):
    if not isinstance(b, (bytes, bytearray)):
        return None
    bio = io.BytesIO(b)
    try:
        bio.seek(0)
        arr = np.load(bio, allow_pickle=True)
        if isinstance(arr, np.ndarray):
            return arr.tolist()
        out = {}
        for kk in arr.files:
            out[kk] = arr[kk].tolist()
        return out
    except Exception:
        return None

_b64_re = re.compile(r'^[A-Za-z0-9+/=\s]+$')
def looks_like_base64_string(s):
    if not isinstance(s, str):
        return False
    if len(s) < 100:
        return False
    ss = ''.join(s.split())
    return bool(_b64_re.fullmatch(ss))

def try_base64_then_numpy_or_json(s):
    if not isinstance(s, str):
        return None
    ss = ''.join(s.split())
    try:
        raw = base64.b64decode(ss)
    except Exception:
        return None
    n = try_numpy_load_bytes(raw)
    if n is not None:
        return n
    try:
        return json.loads(raw.decode('utf-8', errors='replace'))
    except Exception:
        return None

def try_gzip_bytes(b):
    try:
        return gzip.decompress(b)
    except Exception:
        return None

def try_zlib_bytes(b):
    try:
        return zlib.decompress(b)
    except Exception:
        return None

def safe_deserialize(raw):
    """Try many strategies to deserialize Kafka payload."""
    if raw is None:
        return None

    if isinstance(raw, (dict, list, tuple, str, int, float, np.ndarray)):
        return raw

    if isinstance(raw, (bytes, bytearray)):
        j = try_json_bytes(raw)
        if j is not None:
            return j
        
        p = try_pickle_bytes(raw)
        if p is not None:
            return p
        
        n = try_numpy_load_bytes(raw)
        if n is not None:
            return n
        
        g = try_gzip_bytes(raw)
        if g is not None:
            inner = safe_deserialize(g)
            if inner is not None:
                return inner
        
        z = try_zlib_bytes(raw)
        if z is not None:
            inner = safe_deserialize(z)
            if inner is not None:
                return inner
        
        try:
            s = raw.decode('utf-8', errors='replace')
        except Exception:
            s = None
        
        if s:
            j2 = try_json_string(s)
            if j2 is not None:
                return j2
            
            rj = try_relaxed_json_from_string(s)
            if rj is not None:
                return rj
            
            a = try_ast_literal(s)
            if a is not None:
                return a
            
            if looks_like_base64_string(s):
                bnp = try_base64_then_numpy_or_json(s)
                if bnp is not None:
                    return bnp
            
            return s

    try:
        return str(raw)
    except Exception:
        return None


def normalize_weights_payload(payload):
    """Convert payload to list of numpy arrays."""
    if payload is None:
        return None

    if isinstance(payload, str):
        parsed = None
        try:
            parsed = json.loads(payload)
        except Exception:
            try:
                parsed = ast.literal_eval(payload)
            except Exception:
                parsed = None
        if parsed is not None:
            payload = parsed

    if isinstance(payload, dict):
        for key in ('weights', 'model_weights', 'params', 'w', 'payload', 'data'):
            if key in payload:
                payload = payload[key]
                break

    if isinstance(payload, np.ndarray):
        try:
            return [np.array(payload, dtype=np.float32)]
        except Exception:
            return None

    if isinstance(payload, dict) and any(k.startswith('arr_') for k in payload.keys()):
        try:
            ordered_keys = sorted(payload.keys())
            return [np.array(payload[k], dtype=np.float32) for k in ordered_keys]
        except Exception:
            return None

    if isinstance(payload, (list, tuple)):
        try:
            out = []
            for layer in payload:
                if isinstance(layer, (list, tuple, np.ndarray)):
                    arr = np.array(layer, dtype=np.float32)
                elif isinstance(layer, dict) and any(k.startswith('arr_') for k in layer.keys()):
                    ordered_keys = sorted(layer.keys())
                    if len(ordered_keys) == 1:
                        arr = np.array(layer[ordered_keys[0]], dtype=np.float32)
                    else:
                        return None
                else:
                    return None
                out.append(arr)
            return out
        except Exception:
            return None

    return None


# ========================================
# MAIN SERVER
# ========================================
def main():
    """
    Insecure server main loop.
    CRITICAL: Uses same model architecture as secure server.
    """
    consumer = None
    
    try:
        # Load baseline model (SAME as secure server)
        model_path = os.path.join(MODELS_DIR, "ucsd_baseline.h5")
        logger.info(f"Loading baseline model from {model_path}...")
        
        if not os.path.exists(model_path):
            logger.error(f"Baseline model not found at {model_path}")
            logger.error("Please run 'python training/train_baseline_model.py' first!")
            return
        
        try:
            global_model = tf.keras.models.load_model(model_path)
            logger.info("[OK] Baseline model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load baseline model: {e}")
            logger.info("Attempting to load without compile...")
            try:
                global_model = tf.keras.models.load_model(model_path, compile=False)
                global_model.compile(optimizer='adam', loss='mse')
                logger.info("[OK] Model loaded (compile=False)")
            except Exception as e2:
                logger.error(f"All model loading attempts failed: {e2}")
                return

        expected_weights = global_model.get_weights()
        expected_num_layers = len(expected_weights)
        expected_shapes = [w.shape for w in expected_weights]
        
        logger.info(f"Model architecture: {len(global_model.layers)} layers")
        logger.info(f"Expected weight tensors: {expected_num_layers}")
        logger.info(f"Weight shapes: {expected_shapes[:3]}... (showing first 3)")

        # Initialize Kafka consumer
        try:
            consumer = KafkaConsumer(
                KAFKA_TOPIC,
                bootstrap_servers=KAFKA_SERVER,
                value_deserializer=lambda m: m,
                auto_offset_reset='latest',
                enable_auto_commit=True,
                group_id='insecure-server-group'
            )
            logger.info(f"[OK] Connected to Kafka at {KAFKA_SERVER} (topic={KAFKA_TOPIC})")
        except Exception as e:
            logger.error(f"Failed to create Kafka consumer: {e}")
            return

        logger.info("\n" + "="*60)
        logger.info("INSECURE SERVER RUNNING")
        logger.info("Accepts ALL client updates (no security checks)")
        logger.info("="*60 + "\n")

        client_updates = {}
        round_number = 0

        # Duplicate detection
        processed_offsets = set()
        invalid_counts = {}
        ban_list = {}
        INVALID_THRESHOLD = 10
        BAN_SECONDS = 300

        for message in consumer:
            partition = getattr(message, 'partition', None)
            offset = getattr(message, 'offset', None)
            msg_key = message.key
            raw_value = message.value

            # Skip duplicates
            po = (partition, offset)
            if po in processed_offsets:
                logger.debug(f"Skipping duplicate (partition={partition}, offset={offset})")
                continue
            processed_offsets.add(po)

            # Get client ID
            try:
                if isinstance(msg_key, (bytes, bytearray)):
                    key_str = msg_key.decode('utf-8', errors='ignore')
                else:
                    key_str = str(msg_key)
            except Exception:
                key_str = f"unknown-{partition}-{offset}"

            # Check ban
            now_ts = time.time()
            if key_str in ban_list and now_ts < ban_list[key_str]:
                logger.info(f"Skipping banned client {key_str}")
                continue
            elif key_str in ban_list and now_ts >= ban_list[key_str]:
                del ban_list[key_str]
                invalid_counts.pop(key_str, None)

            # Deserialize
            update_data = safe_deserialize(raw_value)
            if update_data is None:
                logger.info(f"[Round {round_number + 1}] Failed to deserialize from {key_str}")
                invalid_counts[key_str] = invalid_counts.get(key_str, 0) + 1
                if invalid_counts[key_str] >= INVALID_THRESHOLD:
                    ban_list[key_str] = time.time() + BAN_SECONDS
                    logger.warning(f"Client {key_str} banned until {datetime.fromtimestamp(ban_list[key_str])}")
                continue

            logger.info(f"\n[Round {round_number + 1}] Received update from: {key_str}")

            # Extract client_id
            client_id = None
            if isinstance(update_data, dict):
                client_id = update_data.get('client_id') or update_data.get('client')
            if not client_id:
                client_id = key_str

            # Extract weights
            weights_payload = None
            if isinstance(update_data, dict):
                for k in ('weights', 'model_weights', 'params', 'w', 'payload', 'data'):
                    if k in update_data:
                        weights_payload = update_data[k]
                        break
                if weights_payload is None and any(k.startswith('arr_') for k in update_data.keys()):
                    weights_payload = update_data
            elif isinstance(update_data, (list, tuple, np.ndarray)):
                weights_payload = update_data

            normalized = normalize_weights_payload(weights_payload)
            
            if normalized is None:
                logger.warning(f"  Invalid weight payload from {client_id}")
                invalid_counts[client_id] = invalid_counts.get(client_id, 0) + 1
                if invalid_counts[client_id] >= INVALID_THRESHOLD:
                    ban_list[client_id] = time.time() + BAN_SECONDS
                continue

            logger.info(f"  Received {len(normalized)} weight layers from {client_id}")
            client_updates[client_id] = normalized
            
            with_weights = sum(1 for v in client_updates.values() if isinstance(v, list) and v)
            logger.info(f"  Updates collected: {with_weights}/{NUM_CLIENTS}")

            # Aggregate when enough updates
            if with_weights >= NUM_CLIENTS:
                round_number += 1
                logger.warning(f"\n{'='*60}")
                logger.warning(f"ROUND {round_number} - INSECURE AGGREGATION")
                logger.warning(f"{'='*60}")

                # Validate shapes
                valid_items = {}
                for cid, wlist in client_updates.items():
                    if not isinstance(wlist, list):
                        logger.warning(f"Skipping {cid}: not a list")
                        continue
                    
                    if len(wlist) != expected_num_layers:
                        logger.warning(f"Skipping {cid}: wrong layer count ({len(wlist)} vs {expected_num_layers})")
                        continue
                    
                    bad_shape = False
                    for idx, layer in enumerate(wlist):
                        got_shape = getattr(layer, 'shape', None)
                        if got_shape != expected_shapes[idx]:
                            logger.warning(f"Skipping {cid}: shape mismatch at layer {idx}")
                            bad_shape = True
                            break
                    
                    if bad_shape:
                        continue
                    
                    try:
                        np_weights = [np.array(layer, dtype=np.float32) for layer in wlist]
                    except Exception:
                        logger.warning(f"Skipping {cid}: numpy conversion failed")
                        continue
                    
                    valid_items[cid] = np_weights

                if not valid_items:
                    logger.error("No valid updates to aggregate. Clearing and continuing.")
                    client_updates = {}
                    continue

                logger.info(f"  Averaging {len(valid_items)} valid updates: {', '.join(valid_items.keys())}")

                # Average weights (INSECURE - no filtering)
                try:
                    averaged_weights = []
                    for layer_idx in range(expected_num_layers):
                        stack = np.stack([valid_items[cid][layer_idx] for cid in valid_items.keys()], axis=0)
                        averaged_layer = np.mean(stack, axis=0)
                        averaged_weights.append(averaged_layer)
                except Exception as e:
                    logger.error(f"Averaging failed: {e}")
                    client_updates = {}
                    continue

                # Update model
                try:
                    global_model.set_weights(averaged_weights)
                    save_path = os.path.join(MODELS_DIR, 'global_model_insecured.h5')
                    
                    # CRITICAL: Use model.save() not save_weights()
                    global_model.save(save_path)
                    logger.info(f"[OK] Model saved to: {save_path}")
                    logger.info(f"     Model has {len(global_model.layers)} layers\n")
                except Exception as e:
                    logger.error(f"Failed to save model: {e}")

                # Reset
                client_updates = {}
                for cid in valid_items.keys():
                    invalid_counts.pop(cid, None)
                    ban_list.pop(cid, None)

                if round_number >= NUM_FEDERATED_ROUNDS:
                    logger.info(f"Completed {NUM_FEDERATED_ROUNDS} rounds - stopping")
                    break

    except KeyboardInterrupt:
        logger.info("\nKeyboard interrupt - stopping server")
    except Exception as e:
        logger.error(f"FATAL ERROR: {e}", exc_info=True)
    finally:
        try:
            if consumer is not None:
                consumer.close()
        except Exception:
            pass
        logger.warning(f"\nInsecure server stopped after {round_number} rounds")


if __name__ == "__main__":
    main()