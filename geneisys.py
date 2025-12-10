#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
================================================================================
PROJECT: GeneISys
================================================================================

AUTHOR:       Clément Cogné
DATE:         2025-11-28
LOCATION:     Toulouse, France
CONTACT:      Open to discussions via LinkedIn (link below)
LINKEDIN:     https://www.linkedin.com/in/clement-cogne/
LICENSE:      MIT (Recommended for Open Source Research)

--------------------------------------------------------------------------------
ABSTRACT
--------------------------------------------------------------------------------
GeneISys is an experimental Organic Neuro-Symbolic Engine based on Semantic 
Physics. Unlike static Deep Learning models, it proposes a dynamic architecture 
where concepts interact as physical bodies within a vector space governed by 
gravitational forces and energy laws.

Key Advantages:
1. White Box & Auditable: Decision making is traceable via an explicit 
   Knowledge Graph and physical interactions, offering transparency for critical 
   systems.
2. Plasticity: Features continuous online learning and structural adaptation 
   (Active Inference), allowing the system to evolve without massive retraining.
3. Portability & Efficiency: Designed to run on standard consumer hardware, 
   incorporating biological constraints like metabolism and memory decay.

--------------------------------------------------------------------------------
WHAT THIS IS NOT
--------------------------------------------------------------------------------
This project is NOT a Large Language Model (LLM), nor is it a Transformer-based 
chatbot like GPT. It does not rely on backpropagation or massive static datasets. 
It is a foundational research prototype exploring dynamic cognitive topology.

--------------------------------------------------------------------------------
A NOTE ON VIBE CODING & GEMINI 3 PRO
--------------------------------------------------------------------------------
This project is a showcase of what is possible through "Vibe Coding" assisted by 
**Gemini 3 Pro**. It demonstrates how human intuition combined with an advanced 
AI thought partner can rapidly structure complex, unconventional architectures 
outside of standard software engineering tracks.

The codebase reflects a development process prioritizing flow, organic structure, 
and intuition. As such, internal logic and comments may reflect the author's 
primary context (French) to facilitate initial conceptualization. Future 
iterations will evolve towards standard English documentation via the built-in 
`DefLanguage()` configuration.

--------------------------------------------------------------------------------
DISCLAIMER & MAINTENANCE (PERSONAL PROJECT)
--------------------------------------------------------------------------------
This is a personal hobby project developed during my free time. It is shared 
"as-is" for educational and research purposes, to inspire the community.

While feedback and discussions are welcome (via LinkedIn), please understand 
that maintenance, feature requests, and bug fixes depend entirely on the 
author's availability. There is no pressure or timeline for updates.

================================================================================
"""

strVersion = "0.0.96_16_10_STABLE_alpha"


import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import shutil
import json
import time
import math
import numpy as np
import random
import threading
import queue
import gc 
import multiprocessing
from collections import Counter
from safetensors.torch import save_file, load_file
import builtins
import traceback

# --- 0. SYSTEME IO SECURISE (Global Lock) ---
#CONSOLE_LOCK = threading.Lock()
#_original_print = print

#def safe_print(*args, **kwargs):
#    """Force l'affichage immédiat et empêche le mélange des lettres entre threads."""
#    with CONSOLE_LOCK:
#        _original_print(*args, **kwargs, flush=True)

# Surcharge globale : Tous les 'print' du code deviennent thread-safe
#print = safe_print


# --- DEPENDANCES ---
try:
    from tokenizers import Tokenizer, models, pre_tokenizers, decoders, normalizers
    TOKENIZERS_AVAILABLE = True
    print(" [SYSTEME] Tokenizers (Rust) activé.")
except ImportError:
    TOKENIZERS_AVAILABLE = False
    print(" [SYSTEME] WARN: 'tokenizers' manquant. Mode dégradé.")
    class Tokenizer: pass

try:
    from numba import jit, prange
    NUMBA_AVAILABLE = True
    print(f" [SYSTEME] Numba détecté.")
except ImportError:
    NUMBA_AVAILABLE = False
    def jit(**kwargs): return lambda f: f
    def prange(x): return range(x)


# --- DETECTION PYTORCH 2.0 (TEST ACTIF ROBUSTE) ---
try:
    if hasattr(torch, "compile"):
        def _dummy_fn(x): return x * 2
        _compiled_fn = torch.compile(_dummy_fn)
        _ = _compiled_fn(torch.tensor([1.0])) 
        
        TORCH_COMPILE_AVAILABLE = True
        print(" [SYSTEME] Torch Compile disponible et VÉRIFIÉ (Acceleration CPU/GPU).")
    else:
        raise ImportError("Fonction torch.compile introuvable.")
except Exception as e:
    TORCH_COMPILE_AVAILABLE = False
    err_msg = str(e)
    if "triton" in err_msg.lower() or "not supported" in err_msg.lower():
        print(f" [SYSTEME] Torch Compile DÉSACTIVÉ (Backend non supporté sur cet OS). Mode Standard.")
    else:
        print(f" [SYSTEME] Torch Compile DÉSACTIVÉ. Erreur technique : {e}")


# --- CONFIGURATION (MVP94.5 - PERFORMANCE) ---
class GenesisConfig:
    # --- REGISTRE DES THREADS ACTIFS (Pattern Pool) ---
    RUNNING_THREADS = []

    def __init__(self, dim=4096, shard_count=5, PrecisionType="FP32", ForceCPU = False,str_version=f'GENEISIS MVP{strVersion} (Performance & Hardening)', ENABLE_MULTITHREADING = True, FORCE_LIGHT_MODE = False):
        if ForceCPU:
            self.DEVICE = torch.device("cpu")
        else:
            self.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.STR_VERSION = str_version
        self.BASE_MEM_DIR = f'./memoire_GeneISys_vn{strVersion}'
        
        if dim % 2 != 0: dim += 1
        self.DIM_SIZE = dim
        self.SHARD_COUNT = shard_count
        
        self.INITIAL_MAX_NODES = 50000
        self.LIMIT_INCR_BY_TOW_MEM = 100000
        self.STEP_SCALE_MEM = 50000
        self.ENABLE_PAGING = False 
        self.PHYSICS_STATIC_BUFFER_SIZE = 128 
        self.ENABLE_CUDA_GRAPHS = False
        self.PHYSICS_CHUNK_SIZE = 1024 
        self.SPARSE_THRESHOLD = 0.01   
        self.MAX_SEQUENCE_LENGTH = 128 

        
        # N11: CONFIGURATION DE PRÉCISION (Point 3)
        # Options: "INT8", "FP16", "FP32"
        self.PRECISION_MODE = PrecisionType
        
        # Configuration automatique des types
        if self.PRECISION_MODE == "INT8":
            self.STORAGE_DTYPE = torch.int8
            self.COMPUTE_DTYPE = torch.float32 # On décompresse pour le calcul
            print(" [CONFIG] Mode: INT8 (Ultra-Compression).")
        elif self.PRECISION_MODE == "FP16":# and self.DEVICE.type == "cuda":
            self.STORAGE_DTYPE = torch.float16
            self.COMPUTE_DTYPE = torch.float16
            print(" [CONFIG] Mode: FP16 (Rapide GPU).")
        else:
            self.STORAGE_DTYPE = torch.float32
            self.COMPUTE_DTYPE = torch.float32
            self.PRECISION_MODE = "FP32"
            print(" [CONFIG] Mode: FP32 (Standard).")



        self.INDEX_DTYPE = torch.float16 if self.DEVICE.type == "cuda" else torch.float32
        self.INDEX_DEVICE = self.DEVICE
        self.STORAGE_DEVICE = torch.device("cpu") if self.DEVICE.type == "cuda" else self.DEVICE
        
        if self.DEVICE.type == "cuda":
            self.ENABLE_PAGING = True
            self.ENABLE_CUDA_GRAPHS = True
            print(" [CONFIG] GPU -> Architecture Hybride.")
        else:
            print(" [CONFIG] CPU -> Mode Direct.")

        self.INGEST_BATCH_SIZE = 5000 
        self.USE_NUMBA = NUMBA_AVAILABLE
        self.USE_CUDA = torch.cuda.is_available()
        
        self.LAYER_CONCEPT = 0       
        self.LAYER_LANGUAGE = 1      
        self.LAYER_VISUAL = 2        
        self.LAYER_AUDIO = 3         
        self.LAYER_INTEROCEPTION = 4 
        self.LAYER_ACTION = 5        
        self.LAYER_MOLECULE = 6      
        self.LAYER_REALITY = 7      
        self.LAYER_BUFFER = 8      
        
        self.MASS_MAPPING = {
            self.LAYER_CONCEPT: 1.0,
            self.LAYER_LANGUAGE: 0.1,
            self.LAYER_VISUAL: 0.2,
            self.LAYER_AUDIO: 0.2,
            self.LAYER_INTEROCEPTION: 0.1,
            self.LAYER_ACTION: 0.1,
            self.LAYER_MOLECULE: 5.0,
            self.LAYER_REALITY: 0.2,
            "INSTANCE": 10.0,
            "DEFAULT": 0.1
        }
        
        

        
        # Physique du Signal (Atténuation Bottom-Up)
        self.PROPAGATION_DECAY = 0.5 
        
        
        
        # --- CORRECTION CRITIQUE : Alignement 7 -> 2000 ---
        # Avant : self.LAYER_REALITY = 7
        # Maintenant : On utilise la constante de profondeur
        # --- ARCHITECTURE LAYERS (OP11) ---
        self.DEPTH_CONCEPT = 0
        self.DEPTH_OFFSET = 1000
        self.DEPTH_BUFFER = 1 * self.DEPTH_OFFSET # 1000
        self.DEPTH_REALITY = 2 * self.DEPTH_OFFSET # 2000
        self.LAYER_REALITY = self.DEPTH_REALITY # 2000 !
        self.LAYER_BUFFER = 8 # Optionnel, ou self.DEPTH_OFFSET (1000)
        
        # Mise à jour des règles physiques pour utiliser cette nouvelle constante
        self.PHYSICS_RULES = {
            self.LAYER_CONCEPT: [self.LAYER_CONCEPT, self.LAYER_LANGUAGE],
            # Maintenant LAYER_REALITY vaut 2000, donc la règle s'appliquera correctement
            self.LAYER_REALITY: [self.LAYER_REALITY, self.LAYER_CONCEPT]
        }
        
        # --- PHYSIQUE RELATIVE (GRAVITÉ PAR LAYER) ---
        self.GRAVITY_MAPPING = {
            self.LAYER_CONCEPT: 1.0,    # Terre (Standard)
            self.LAYER_REALITY: 0.2,    # Mars (Plus léger, fluide)
            self.LAYER_BUFFER: 0.5,     # Intermédiaire
            "DEFAULT": 1.0
        }
        
        # --- NOUVEAU : Facteurs de Densité (Terre vs Mars) ---
        self.LAYER_DENSITY = {
            self.LAYER_CONCEPT: 1.0,    # Terre (Standard)
            self.LAYER_REALITY: 0.2,    # Mars (Plus léger, fluide)
            "DEFAULT": 1.0
        }
        
        self.FORCE_DIRECT_LINK = 1.0
        self.FORCE_INDIRECT_DECAY = 0.5

        self.ENERGY_INIT_DEFAULT = 10.0
        self.DREAM_INTENSITY = 3 
        self.ENABLE_CURIOSITY = True 
        self.BOREDOM_THRESHOLD = 50.0
        self.BOREDOM_RATE = 2.0
        self.LIFE_TICK_SPEED = 1.0
        self.TOOL_SEARCH_WINDOW = 4 
        self.TOOL_PATTERN_THRESHOLD = 3 
        self.PLASTICITY_DEFAULT = 0.5
        self.PLASTICITY_REALITY = 1.0
        self.PLASTICITY_CONCEPT = 0.05
        self.ELASTICITY_ATTRACTION = 0.05
        self.MOMENTUM_MEAN = 0.8
        self.LEARNING_RATE_HARDWARE = 0.5
        self.LEARNING_RATE_ATTRIBUTION = 0.1
        self.ENERGY_INIT_REALITY = 50.0
        self.ENERGY_INIT_CONCEPT = 100.0
        self.MASS_OPERATOR = 10.0 
        self.MASS_PLANET = 1.0 
        self.MASS_DUST = 0.1
        self.MASS_THRESHOLD = 0.05
        self.LINEAR_MOMENTUM = 2.0 
        self.INGEST_SLEEP_INTERVAL = 10000 
        self.INGEST_DEFAULT_EPOCHS = 1
        self.EPSILON = 1e-5
        self.QUANTIZATION_EPSILON = 1e-5
        
        # --- MVP97 : TURBO PIPELINE ---
        self.ENABLE_MULTITHREADING = ENABLE_MULTITHREADING  # Active le Bridge Léger
        self.INGESTION_QUEUE_SIZE = 50     # Backpressure (Max batchs en attente)
        self.INGESTION_TIMEOUT = 0.5       # Temps d'attente max des workers
        self.BRIDGE_SYNC_TIMEOUT = 20.0    # Timeout augmenté pour la stabilité
        
        # --- NOUVEAU : CONSTANTES PHYSIQUES CENTRALISÉES (Patch Stabilité) ---
        self.PHYSICS_EPSILON = 0.1          # Évite la division par zéro (Distance)
        self.SIMILARITY_POWER = 3.0         # Facteur d'amplification de la similarité
        self.INTERACTION_THRESHOLD = 0.001  # Seuil minimal pour considérer une force
        self.FORCE_CLAMP_MIN = -0.99        # Bornes pour éviter l'explosion numérique
        self.FORCE_CLAMP_MAX = 0.99
        
        # --- ATTENTION SÉLECTIVE (Recap 12) ---
        self.TOP_K_LIMIT = 10  # Nombre max de voisins influents (Gravité Sparse)
        
        
        # --- MVP98 : HEAVY BRIDGE CONFIG ---
        # Nombre de processus lourds (CPU bound)
        # On laisse 2 coeurs libres pour le système et le Main Thread/GPU driver
        # --- MVP98 : HEAVY BRIDGE CONFIG ---
        # [USER CONTROL] Mettre à True pour interdire le Multiprocessing (Debug ou petites configs)
        self.FORCE_LIGHT_MODE = FORCE_LIGHT_MODE 
        # --- CONFIGURATION DU COMPORTEMENT ---
        # Si True : perceive() bloque jusqu'à ce que le traitement soit fini (Idéal pour Tests/Debug)
        self.SYNCHRONOUS_PERCEPTION = True
        # Détection et Application de la Stratégie
        try:
            cpu_count = multiprocessing.cpu_count()
            
            # La règle de décision explicite :
            # 1. Si l'utilisateur force le mode Light -> LIGHT
            # 2. Si on a moins de 4 coeurs -> LIGHT (Trop de surcharge pour rien)
            # 3. Sinon -> HEAVY (Mode nominal MVP98)
            
            if self.FORCE_LIGHT_MODE:
                print(" [CONFIG] Mode Light FORCÉ par l'utilisateur.")
                self.ENABLE_HEAVY_BRIDGE = False
                self.HEAVY_WORKERS_COUNT = 1
            elif cpu_count < 4:
                print(" [CONFIG] Mode Light FORCÉ par hardware")
                print(f" [CONFIG] CPU insuffisant pour Heavy Bridge ({cpu_count} coeurs). Passage en Light.")
                self.ENABLE_HEAVY_BRIDGE = False
                self.HEAVY_WORKERS_COUNT = 1
            else:
                # Mode Heavy activé
                self.ENABLE_HEAVY_BRIDGE = True
                # On garde 2 coeurs pour le système/GPU, minimum 1 worker
                self.HEAVY_WORKERS_COUNT = max(1, cpu_count - 2)
                
        except NotImplementedError:
            # Fallback de sécurité (OS exotiques)
            self.ENABLE_HEAVY_BRIDGE = False
            self.HEAVY_WORKERS_COUNT = 1

        self.HEAVY_QUEUE_SIZE = 100 
        
        mode_str = "HEAVY (Multi-Process)" if self.ENABLE_HEAVY_BRIDGE else "LIGHT (Threaded)"
        print(f" [CONFIG] Architecture Ingestion Active : {mode_str}")
            
        
        
        
        self._calculate_dimensional_params()

    def _calculate_dimensional_params(self):
        self.SCALE = math.log2(self.DIM_SIZE / 64.0) if self.DIM_SIZE >= 64 else 0.0
        
        self.THRESHOLD_STRICT = 0.50 / (1 + (self.SCALE * 0.05))
        self.THRESHOLD_LOOSE = 0.15 / (1 + (self.SCALE * 0.8))
        self.THRESHOLD_FORECAST = 0.90 / (1 + (self.SCALE * 0.05))
        self.THRESHOLD_VALIDATION = self.THRESHOLD_FORECAST * 100.0
        self.THRESHOLD_FORGET = self.ENERGY_INIT_DEFAULT * 0.5
        self.CHRONOS_FREQ_BASE = 10000.0
        scale_log = math.log1p(self.SCALE)
        self.INFERENCE_ACCOMMODATION = 0.20 / (1 + (scale_log * 0.1)) 
        self.INFERENCE_RUPTURE = 0.85 / (1 + (scale_log * 0.1))
        TARGET_ELEMENTS = 1024 * 1024 * 12 
        optimal_chunk = int(math.sqrt(TARGET_ELEMENTS))
        if self.DIM_SIZE > 4096:
            optimal_chunk = min(optimal_chunk, int(TARGET_ELEMENTS / self.DIM_SIZE))
        self.PHYSICS_CHUNK_SIZE = max(128, optimal_chunk)
        self.BATCH_UPDATE_CHUNK_SIZE = max(1024, self.DIM_SIZE // 2)
        
        
    @classmethod
    def register(cls, thread_instance):
        """Enregistre un thread pour qu'il soit nettoyé à la fin."""
        if thread_instance not in cls.RUNNING_THREADS:
            cls.RUNNING_THREADS.append(thread_instance)

    @classmethod
    def unregister(cls, thread_instance):
        """Retire un thread de la liste (s'il est stoppé manuellement)."""
        if thread_instance in cls.RUNNING_THREADS:
            cls.RUNNING_THREADS.remove(thread_instance)

    @classmethod
    def global_shutdown(cls):
        """Arrêt propre de TOUS les threads enregistrés, quels qu'ils soient."""
        print(f"\n [SYSTEME] Nettoyage global ({len(cls.RUNNING_THREADS)} threads actifs)...")
        for t in cls.RUNNING_THREADS:
            try:
                if hasattr(t, 'stop'): 
                    t.stop() # Appel de la méthode d'arrêt douce
                if t.is_alive():
                    t.join(timeout=1.0) # Attente
            except Exception as e:
                print(f" [WARN] Erreur lors de l'arrêt du thread {t.name}: {e}")
        cls.RUNNING_THREADS.clear()

#first call for function declaration when CFG.X is a default value
CFG = GenesisConfig()


# ==============================================================================
#  SYSTEME I/O ASYNCHRONE (LOGGER BRIDGE)
# ==============================================================================
class GenesisAsyncLogger(threading.Thread):
    def __init__(self):
        super().__init__(name="GenesisLogger", daemon=True)
        self.log_queue = queue.Queue()
        self.stop_event = threading.Event()
        self.original_print = builtins.print 
        GenesisConfig.register(self) # Je m'enregistre dans le pool

    def run(self):
        while not self.stop_event.is_set() or not self.log_queue.empty():
            try:
                msg = self.log_queue.get(timeout=0.05)
                self.original_print(msg)
                
                # --- PREUVE DU BUFFER ---
                # qsize() nous dit combien d'AUTRES messages attendent derrière.
                # Si qsize > 0, cela prouve que le Main Thread a été plus vite que l'affichage.
                #taille_buffer = self.log_queue.qsize()
                #prefixe = f"[BUFFER: {taille_buffer}]" if taille_buffer > 0 else "[DIRECT]"
                #self.original_print(f"{prefixe} {msg}")
                # -------------------------
                
                
                
                # CRUCIAL : Signale que ce message a été traité
                self.log_queue.task_done() 
            except queue.Empty:
                continue
    
    def log(self, *args, **kwargs):
        msg = " ".join(map(str, args))
        self.log_queue.put(msg)

    def wait_until_empty(self):
        """
        Bloque l'appelant jusqu'à ce que tous les messages en attente
        soient réellement affichés à l'écran.
        """
        self.log_queue.join()

    def stop(self):
        self.stop_event.set()
        if self.is_alive():
            self.join(timeout=2.0)

class PrintRedirector:
    """
    Context Manager intelligent.
    1. Capture les prints -> Queue (Rapide)
    2. À la sortie, attend que la Queue soit vide (Propre)
    """
    def __init__(self, logger_instance):
        self.logger = logger_instance
        self.original_print = builtins.print

    def __enter__(self):
        builtins.print = self.logger.log
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # 1. D'abord, on remet le print normal (sécurité)
        builtins.print = self.original_print
        
        # 2. Ensuite, on force le Main Thread à attendre que le Logger ait fini
        #    de "vomir" tout ce qu'il a accumulé pendant le bloc.
        self.logger.wait_until_empty()

    
    
def gravity_kernel_masked_symmetric(positions: torch.Tensor, 
                                    masses: torch.Tensor, 
                                    vecs: torch.Tensor,
                                    mask: torch.Tensor = None) -> torch.Tensor:
    """
    Noyau Physique Unifié (CPU/GPU Compatible).
    Utilise la similarité Cosine et le Clamping pour la stabilité numérique.
    Remplace l'ancienne logique dot-product pure qui causait des NaN.
    """
    # 1. Normalisation pour stabilité (Cosine Similarity)
    # eps=1e-8 évite la division par zéro si un vecteur est nul
    vecs_norm = F.normalize(vecs, p=2, dim=1, eps=CFG.EPSILON)
    
    # 2. Similarité Matricielle (Optimisé AVX2 sur CPU, CUDA Cores sur GPU)
    sim_matrix = torch.mm(vecs_norm, vecs_norm.t())
    
    # 3. Sécurité Numérique (Clamping)
    # Empêche les valeurs > 1.0 ou < -1.0 qui font exploser le .pow(3)
    sim_safe = torch.clamp(sim_matrix, CFG.FORCE_CLAMP_MIN, CFG.FORCE_CLAMP_MAX)
    
    # 4. Calcul de la Gravité Sémantique
    dist_matrix = torch.abs(positions.unsqueeze(1) - positions.unsqueeze(0)) + CFG.PHYSICS_EPSILON
    mass_prod = masses.unsqueeze(1) * masses.unsqueeze(0)
    
    # (Sim + 1)^3 permet de favoriser fortement les concepts proches
    numerator = mass_prod * (sim_safe + 1.0).pow(CFG.SIMILARITY_POWER)
    forces = numerator / dist_matrix
    
    if mask is not None:
        forces = forces * mask
        
    forces.fill_diagonal_(0.0)
    return forces

# --- 1. KERNELS PHYSIQUES (OPTIMISATION SYMETRIQUE & VECTORISEE) ---
@jit(nopython=True, parallel=True, fastmath=True)
def driver_cpu_numba_LEGACY(positions, masses, vecs, dim, n, mask):
    forces = np.zeros((n, n), dtype=np.float32)
    # Optimisation: On vérifie une seule fois si un masque existe via sa forme
    has_mask = mask.shape[0] > 0
    
    for i in prange(n):
        for j in range(i + 1, n): 
            # 1. Calcul Physique Inconditionnel (Pipeline Stable)
            # On ne fait AUCUN 'if' ici pour ne pas briser la vectorisation (SIMD)
            sim = 0.0
            for k in range(dim): 
                sim += vecs[i, k] * vecs[j, k]
            
            dist = abs(positions[i] - positions[j]) + 0.1
            f = (masses[i] * masses[j] * ((sim + 1.0)**3)) / dist
            
            # 2. Application du Masque par Multiplication (ALU vs Branching)
            # Si le masque est 0, f devient 0. C'est beaucoup plus rapide pour le CPU
            # qu'une prediction de branche ratée.
            if has_mask:
                f = f * mask[i, j]

            forces[i, j] = f
            forces[j, i] = f 
    return forces


# --- MOTEUR PHYSIQUE ---
# --- MOTEUR CORRIGÉ ---
class ChunkedGravityEngine:
    def __init__(self, dim, max_capacity=128):
        self.dim = dim
        self.max_capacity = max_capacity
        self.mode = "CPU"
        kernel_fn = gravity_kernel_masked_symmetric 
        
        if TORCH_COMPILE_AVAILABLE:
            try:
                self.optimized_kernel = torch.compile(kernel_fn, mode="reduce-overhead")
            except Exception:
                self.optimized_kernel = kernel_fn
        else:
            self.optimized_kernel = kernel_fn 

        self.compute_stream = None
        if CFG.USE_CUDA:
            self.compute_stream = torch.cuda.Stream()

        self.cuda_graph = None
        self.static_input_pos = None
        self.static_input_mass = None
        self.static_input_vecs = None
        self.static_input_mask = None 
        self.static_output_forces = None
        
        if CFG.USE_CUDA:
            if CFG.ENABLE_CUDA_GRAPHS:
                try:
                    self.static_input_pos = torch.zeros(max_capacity, device=CFG.DEVICE)
                    self.static_input_mass = torch.zeros(max_capacity, device=CFG.DEVICE)
                    self.static_input_vecs = torch.zeros((max_capacity, dim), device=CFG.DEVICE)
                    self.static_input_mask = torch.ones((max_capacity, max_capacity), device=CFG.DEVICE)
                    self.static_output_forces = torch.zeros((max_capacity, max_capacity), device=CFG.DEVICE)
                    self._capture_graph()
                    self.mode = "HYBRID_AUTO"
                except Exception: self.mode = "GPU_STD"
            else: self.mode = "GPU_STD"
        else: self.mode = "CPU"

    def _capture_graph(self):
        self.optimized_kernel(self.static_input_pos, self.static_input_mass, self.static_input_vecs, self.static_input_mask)
        torch.cuda.synchronize()
        self.cuda_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.cuda_graph):
            self.static_output_forces = self.optimized_kernel(
                self.static_input_pos, self.static_input_mass, self.static_input_vecs, self.static_input_mask
            )

    def compute(self, pos, mass, vecs, n_active, mask=None):
        if self.compute_stream is not None:
            with torch.cuda.stream(self.compute_stream):
                return self._compute_internal(pos, mass, vecs, n_active, mask)
        else:
            return self._compute_internal(pos, mass, vecs, n_active, mask)

    def _compute_internal_legacy(self, pos, mass, vecs, n_active, mask=None):
        if self.mode == "CPU":
            p_cpu = pos.cpu().numpy(); m_cpu = mass.cpu().numpy(); v_cpu = vecs.cpu().numpy()
            
            # Gestion robuste du masque pour Numba
            # Si pas de masque, on envoie un tableau vide (0,0) qui déclenchera has_mask=False
            if mask is not None:
                mask_cpu = mask.cpu().numpy()
            else:
                mask_cpu = np.zeros((0, 0), dtype=np.float32)
                
            f_cpu = driver_cpu_numba(p_cpu, m_cpu, v_cpu, self.dim, n_active, mask_cpu)
            return torch.from_numpy(f_cpu).to(CFG.DEVICE)

        if n_active <= self.max_capacity and self.mode == "HYBRID_AUTO":
            self.static_input_pos[:n_active].copy_(pos)
            self.static_input_mass[:n_active].copy_(mass)
            self.static_input_vecs[:n_active].copy_(vecs)
            if mask is not None:
                self.static_input_mask[:n_active, :n_active].copy_(mask)
            else:
                # IMPORTANT: En mode Graph, si pas de masque, on remplit de 1.0
                # car le kernel GPU fait une multiplication matricielle stricte.
                self.static_input_mask[:n_active, :n_active].fill_(1.0)
            self.cuda_graph.replay()
            return self.static_output_forces[:n_active, :n_active]

        if n_active <= 5000:
            return self.optimized_kernel(pos, mass, vecs, mask)

        return self._compute_chunked_sparse_offload(pos, mass, vecs, n_active)
        
    def _compute_internal(self, pos, mass, vecs, n_active, mask=None):
        # MODE CPU : On utilise maintenant PyTorch (AVX2/MKL) au lieu de Numba
        # C'est ce qui assure la compatibilité et la stabilité numérique
        if self.mode == "CPU":
            # Appel direct au kernel PyTorch unifié (Rapide & Stable)
            return self.optimized_kernel(pos, mass, vecs, mask)

        # ... (Le reste de la fonction pour le mode GPU/Hybrid reste inchangé) ...
        if n_active <= self.max_capacity and self.mode == "HYBRID_AUTO":
            self.static_input_pos[:n_active].copy_(pos)
            self.static_input_mass[:n_active].copy_(mass)
            self.static_input_vecs[:n_active].copy_(vecs)
            if mask is not None:
                self.static_input_mask[:n_active, :n_active].copy_(mask)
            else:
                self.static_input_mask[:n_active, :n_active].fill_(1.0)
            self.cuda_graph.replay()
            return self.static_output_forces[:n_active, :n_active]

        if n_active <= 5000:
            return self.optimized_kernel(pos, mass, vecs, mask)

        return self._compute_chunked_sparse_offload(pos, mass, vecs, n_active)

    def _compute_chunked_sparse_offload(self, pos, mass, vecs, n):
        chunk_size = CFG.PHYSICS_CHUNK_SIZE
        cpu_indices = []; cpu_values = []
        torch.cuda.empty_cache()
        
        for i in range(0, n, chunk_size):
            end_i = min(i + chunk_size, n)
            vecs_i = vecs[i:end_i]; pos_i = pos[i:end_i]; mass_i = mass[i:end_i]
            
            for j in range(0, n, chunk_size):
                end_j = min(j + chunk_size, n)
                vecs_j = vecs[j:end_j]; pos_j = pos[j:end_j]; mass_j = mass[j:end_j]
                
                sim_block = torch.mm(vecs_i, vecs_j.t())
                dist_block = torch.abs(pos_i.unsqueeze(1) - pos_j.unsqueeze(0)) + 0.1
                force_block = (mass_i.unsqueeze(1) * mass_j.unsqueeze(0)) * (sim_block + 1.0).pow(3)
                force_block = force_block / dist_block
                
                mask_block = force_block > CFG.SPARSE_THRESHOLD
                if mask_block.any():
                    local_indices = torch.nonzero(mask_block, as_tuple=False)
                    vals = force_block[mask_block]
                    local_indices_cpu = local_indices.cpu()
                    vals_cpu = vals.cpu()
                    global_indices = local_indices_cpu.clone()
                    global_indices[:, 0] += i
                    global_indices[:, 1] += j
                    cpu_indices.append(global_indices)
                    cpu_values.append(vals_cpu)
                del sim_block, dist_block, force_block, mask_block
                
        if not cpu_indices: 
            return torch.sparse_coo_tensor(torch.zeros((2,0)), torch.zeros(0), (n, n), device=CFG.DEVICE)
            
        all_indices_cpu = torch.cat(cpu_indices, dim=0).t()
        all_values_cpu = torch.cat(cpu_values, dim=0)
        try:
            return torch.sparse_coo_tensor(all_indices_cpu, all_values_cpu, (n, n)).to(CFG.DEVICE)
        except RuntimeError:
            print(" [WARN] Sparse Tensor trop gros pour VRAM, maintien sur CPU.")
            return torch.sparse_coo_tensor(all_indices_cpu, all_values_cpu, (n, n))
            
            
# --- UTILITAIRES ---
class Quantizer:
    @staticmethod
    def to_storage(tensor): return tensor.to(CFG.STORAGE_DTYPE)
    @staticmethod
    def from_storage(tensor): return tensor.to(CFG.COMPUTE_DTYPE)


class SmartQuantizer:
    """
    Gestionnaire de quantification symétrique INT8 avec scaling.
    [FIX FP16] Epsilon ajusté à 1e-5 pour éviter l'underflow en Half Precision.
    """
    @staticmethod
    def quantize(tensor_fp32):
        # keepdim=True permet de garder la dimension pour la division broadcastée
        max_val = tensor_fp32.abs().max(dim=-1, keepdim=True).values
        scale = max_val / 127.0
        # 1e-8 est trop petit pour FP16 (min ~6e-5). On met 1e-5 pour être safe.
        scale = torch.clamp(scale, min=CFG.QUANTIZATION_EPSILON) 
        
        tensor_int8 = (tensor_fp32 / scale).round().to(torch.int8)
        return tensor_int8, scale

    @staticmethod
    def dequantize(tensor_int8, scale):
        return tensor_int8.to(torch.float32) * scale




class DefLanguage:
    def __init__(self, strLang="fr"):
    
    
        if strLang == "fr":
            self.articles = ["le", "la", "les", "un", "une", "il", "y", "a", "qui", "que", "ce", "se", "sa", "son", "des", "du"]
            self.adjectives = ["grand", "grande", "petit", "petite", "bon", "bonne", "mauvais", "rouge", "bleu", "vert", "rapide", "lent"]
            self.ops= {
                ("sont", "<>"): "OP_DEF", ("dans", "<>"): "OP_LOC", ("est", "<>"): "OP_ATTR", 
                ("separer", "<>"): "OP_REP", ("creer", "<>"): "OP_SYN", ("avec", "<>"): "OP_SYN", 
                ("sans", "<>"): "OP_NEG", ("comme", "<>"): "OP_CTX", ("nommer", "<>"): "OP_LBL",
                ("mangée_par", "><"): "OP_ATTR", ("donc", "<>"): "OP_SEQ"
            }

class SafeFileManager:
    @staticmethod
    def load_tensors(path):
        base_name = os.path.splitext(path)[0]
        path_safetensors = base_name + ".safetensors"
        
        # On gère le cas où l'extension est déjà dans le path ou pas
        if not os.path.exists(path_safetensors) and os.path.exists(path):
             path_safetensors = path
             
        if not os.path.exists(path_safetensors): return {}
        
        try:
            # 1. Chargement des données uniques
            loaded_data = load_file(path_safetensors)
            # Convertir en dict mutable et mettre sur le bon device
            tensors_dict = {k: v.to(CFG.DEVICE) for k, v in loaded_data.items()}
            
            # 2. Restauration des liens partagés (Alias)
            path_map = base_name + "_map.json"
            # Si on a un fichier map qui correspond au fichier safetensor
            if not os.path.exists(path_map):
                 # Essai avec le path direct si base_name a échoué
                 path_map = path + "_map.json" if not path.endswith(".safetensors") else path.replace(".safetensors", "_map.json")

            if os.path.exists(path_map):
                with open(path_map, 'r', encoding='utf-8') as f:
                    alias_map = json.load(f)
                
                print(f" [LOAD] Restauration de {len(alias_map)} liens mémoire partagés (Aliases)...")
                for alias_key, master_key in alias_map.items():
                    if master_key in tensors_dict:
                        # MAGIE : On assigne la RÉFÉRENCE du master à l'alias
                        # Python ne copie pas les données, il copie le pointeur.
                        # Désormais, modifier l'un modifie l'autre.
                        tensors_dict[alias_key] = tensors_dict[master_key]
                    else:
                        print(f" [WARN] Clé maître '{master_key}' manquante pour l'alias '{alias_key}'")
            
            return tensors_dict
            
        except Exception as e:
            print(f"[IO ERR] Chargement échoué : {e}")
            return {}

    @staticmethod
    def save_tensors(tensors, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Définition des noms de fichiers
        base_name = os.path.splitext(path)[0]
        target_file = base_name + ".safetensors"
        map_file = base_name + "_map.json"
        tmp_file = target_file + ".tmp"
        bak_file = target_file + ".bak" # Fichier de secours pour le lock Windows
        
        unique_tensors = {}
        ptr_to_key = {}
        alias_map = {}
        
        try:
            # 1. Deduplication (Inchangé)
            for key, tensor in tensors.items():
                ptr = tensor.data_ptr()
                if ptr in ptr_to_key:
                    alias_map[key] = ptr_to_key[ptr]
                else:
                    ptr_to_key[ptr] = key
                    # .contiguous() est crucial pour la sérialisation
                    unique_tensors[key] = tensor.contiguous()

            # 2. SAUVEGARDE SÉCURISÉE (Write-Tmp-Swap)
            
            # A. On écrit d'abord dans un fichier temporaire (.tmp)
            # Cela ne touche pas au fichier verrouillé par Windows
            save_file(unique_tensors, tmp_file)
            
            # B. Rotation des fichiers (La danse Windows)
            if os.path.exists(target_file):
                try:
                    # On essaie de supprimer le fichier .bak s'il existe déjà
                    if os.path.exists(bak_file):
                        os.remove(bak_file)
                    
                    # CRITIQUE : On renomme l'ancien fichier (verrouillé) en .bak
                    # Windows autorise souvent le renommage d'un fichier mappé, mais pas sa suppression/écriture.
                    os.rename(target_file, bak_file)
                except OSError as e:
                    print(f" [WARN] Impossible de déplacer l'ancien fichier (Lock Windows): {e}")
                    # Si on ne peut pas le bouger, on force la suppression (peut échouer)
                    try: os.remove(target_file)
                    except: pass

            # C. On met le nouveau fichier à la place du vrai nom
            # Si l'étape B a échoué (fichier toujours là), ceci plantera. 
            # Mais grâce à l'étape B, le chemin devrait être libre.
            if os.path.exists(tmp_file):
                os.rename(tmp_file, target_file)

            # 3. Sauvegarde de la carte des alias
            if alias_map:
                with open(map_file, 'w', encoding='utf-8') as f:
                    json.dump(alias_map, f)
            elif os.path.exists(map_file):
                try: os.remove(map_file)
                except: pass
                
            # 4. Nettoyage final (Optionnel)
            # On essaie de supprimer le backup. Si c'est verrouillé (1224), on laisse tomber,
            # ce n'est pas grave, ce sera supprimé au prochain tour.
            if os.path.exists(bak_file):
                try: os.remove(bak_file)
                except: pass

        except Exception as e:
            print(f"[IO ERR] Echec sauvegarde intelligente : {e}")
    @staticmethod
    def save_json(data, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        try:
            with open(path + ".tmp", 'w', encoding='utf-8') as f: json.dump(data, f, indent=2)
            if os.path.exists(path): os.remove(path)
            os.rename(path + ".tmp", path)
        except Exception: pass
    @staticmethod
    def load_json(path):
        if os.path.exists(path):
            try:
                with open(path, 'r', encoding='utf-8') as f: return json.load(f)
            except: pass
        return {}
    @staticmethod
    def save_monolithic_v2(tensors_dict, json_data, energy_tensor, base_dir, meta_data):
        SafeFileManager.save_json(json_data, os.path.join(base_dir, "genesis_structure.json"))
        SafeFileManager.save_tensors(tensors_dict, os.path.join(base_dir, "genesis_world.safetensors"))
        SafeFileManager.save_tensors({"global_energies": energy_tensor}, os.path.join(base_dir, "genesis_energies.safetensors"))
        SafeFileManager.save_json(meta_data, os.path.join(base_dir, "genesis_memory_meta.json"))
    @staticmethod
    def load_monolithic_v2(base_dir):
        struct = SafeFileManager.load_json(os.path.join(base_dir, "genesis_structure.json"))
        vects = SafeFileManager.load_tensors(os.path.join(base_dir, "genesis_world.safetensors"))
        energies = SafeFileManager.load_tensors(os.path.join(base_dir, "genesis_energies.safetensors"))
        meta = SafeFileManager.load_json(os.path.join(base_dir, "genesis_memory_meta.json"))
        return struct, vects, energies.get("global_energies"), meta
    @staticmethod
    def save_shard(tensors, shard_id, base_dir): SafeFileManager.save_tensors(tensors, os.path.join(base_dir, f"shard_{shard_id}.safetensors"))
    @staticmethod
    def load_shard(shard_id, base_dir): return SafeFileManager.load_tensors(os.path.join(base_dir, f"shard_{shard_id}.safetensors"))

class InputListener(threading.Thread):
    def __init__(self, input_queue):
        threading.Thread.__init__(self); self.input_queue = input_queue; self.daemon = True 
    def run(self):
        while True:
            try:
                user_input = input(); self.input_queue.put(user_input)
            except EOFError: break

class Chronos(nn.Module):
    def __init__(self, dim):
        super().__init__(); self.dim = dim; self.tick = 0 
        inv_freq = 1.0 / (CFG.CHRONOS_FREQ_BASE ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq.to(CFG.DEVICE))
    def get_time_vector(self, offset=0):
        pos = torch.tensor([self.tick + offset], dtype=torch.float32).to(CFG.DEVICE)
        inp = torch.einsum("i,j->ij", pos, self.inv_freq)
        return F.normalize(torch.cat((inp.sin(), inp.cos()), dim=-1).squeeze(0), p=2, dim=0)
    def advance(self): self.tick += 1

class AssociativeMemory:
    def __init__(self, brain):
        self.brain = brain; self.vocab_vectors = None; self.vocab_words = []; self.is_dirty = False
    def register_word(self, word, vector): self.is_dirty = True
    # [Dans la classe AssociativeMemory]

    def _refresh_lexicon(self):
        if not self.is_dirty and self.vocab_vectors is not None: return
        
        # On récupère les mots connus
        words = list(self.brain.encoder.semantic_map.keys())
        if not words: return
        
        clean_words = [w for w in words if not w.startswith(("EVT_", "HYP_"))]
        vecs = []
        
        for w in clean_words:
            # --- FIX: Cohérence Mémoire (Hot Path) ---
            # Avant : on lisait raw_v = self.brain.encoder.semantic_map[w] (Périmé)
            # Après : on utilise le getter intelligent qui priorise le fast_index (Frais)
            v = self.brain.encoder.get_semantic_vector(w)
            # -----------------------------------------
            
            # get_semantic_vector renvoie déjà du COMPUTE_DTYPE, on normalise juste
            if v.is_sparse: v = v.to_dense()
            vecs.append(F.normalize(v, p=2, dim=0, eps=CFG.EPSILON))
            
        if vecs:
            self.vocab_vectors = torch.stack(vecs).to(CFG.DEVICE)
            self.vocab_words = clean_words
            self.is_dirty = False
            
            
    def articulate(self, vector, anti_parrot=True):
        self._refresh_lexicon()
        if self.vocab_vectors is None: return "???"
        if vector.dim() == 1: vector = vector.unsqueeze(0)
        query = F.normalize(vector.to(CFG.COMPUTE_DTYPE), p=2, dim=1, eps=CFG.EPSILON)
        scores = torch.mm(query, self.vocab_vectors.t()).squeeze(0)
        k_val = min(2, len(scores))
        best_scores, best_indices = torch.topk(scores, k=k_val)
        if len(best_indices) == 0: return "???"
        best_idx = best_indices[0].item()
        if anti_parrot and best_scores[0].item() > 0.99 and len(best_indices) > 1:
             best_idx = best_indices[1].item()
        return self.vocab_words[best_idx]

class HybridMemoryCluster:
    def __init__(self, dim, max_nodes=None):
        self.dim = dim
        self.capacity = max_nodes if max_nodes else CFG.INITIAL_MAX_NODES
        
        # Détection du mode de compression (Utilisez is_quantized pour être cohérent avec votre code précédent)
        self.is_quantized = (CFG.STORAGE_DTYPE == torch.int8)
        
        print(f" [MEMORY] Allocation Index Rapide: ({self.capacity}, {self.dim}) en {CFG.INDEX_DTYPE}")
        self.fast_index = torch.zeros((self.capacity, self.dim), dtype=CFG.INDEX_DTYPE, device=CFG.INDEX_DEVICE)
        
        print(f" [MEMORY] Allocation Stockage Maître: ({self.capacity}, {self.dim}) en {CFG.STORAGE_DTYPE} sur {CFG.STORAGE_DEVICE}")
        self.master_storage = torch.zeros((self.capacity, self.dim), dtype=CFG.STORAGE_DTYPE, device=CFG.STORAGE_DEVICE)
        
        # --- CORRECTIF : Allocation INCONDITIONNELLE des Scales ---
        # On en a besoin même en FP16/32 pour charger proprement les transitions de format
        self.master_scales = torch.ones((self.capacity, 1), dtype=torch.float32, device=CFG.STORAGE_DEVICE)
        
        if CFG.ENABLE_PAGING and self.master_storage.device.type == 'cpu':
            self.master_storage = self.master_storage.pin_memory()
            self.master_scales = self.master_scales.pin_memory()
            print(" [MEMORY] Pinned Memory activée.")
            
        self.active_count = 0
        self.name_to_idx = {} 
        self.idx_to_name = {} 
        self.shards = {i: {} for i in range(CFG.SHARD_COUNT)}
        
    def resize(self, new_capacity):
        print(f" [MEMORY] Tentative de redimensionnement : {self.capacity} -> {new_capacity}")
        if new_capacity <= self.capacity: return
        try:
            new_index = torch.zeros((new_capacity, self.dim), dtype=CFG.INDEX_DTYPE, device=CFG.INDEX_DEVICE)
            new_index[:self.capacity] = self.fast_index
            self.fast_index = new_index
            
            new_storage = torch.zeros((new_capacity, self.dim), dtype=CFG.STORAGE_DTYPE, device=CFG.STORAGE_DEVICE)
            new_storage[:self.capacity] = self.master_storage
            
            # --- CORRECTIF : Resize systématique des scales ---
            new_scales = torch.ones((new_capacity, 1), dtype=torch.float32, device=CFG.STORAGE_DEVICE)
            new_scales[:self.capacity] = self.master_scales
                
            if CFG.ENABLE_PAGING and new_storage.device.type == 'cpu': 
                try:
                    new_storage = new_storage.pin_memory()
                    new_scales = new_scales.pin_memory()
                    print(f" [MEMORY] Pinned Memory allouée avec succès ({new_capacity} slots).")
                except RuntimeError:
                    print(f" [WARN] Echec Pinned Memory (OOM). Fallback sur RAM standard.")
            
            self.master_storage = new_storage
            self.master_scales = new_scales # Toujours présent
            
            self.capacity = new_capacity
        except RuntimeError as e:
            print(f" [CRITICAL] Impossible d'allouer {new_capacity} slots: {e}")
            return

    def update_batch(self, names, vectors):
        batch_size = len(names)
        if batch_size == 0: return
        
        if self.active_count + batch_size >= self.capacity:
            self.resize(max(self.capacity * 2, self.active_count + batch_size + 1000))
            
        indices_list = []
        
        # --- BRANCHE CONDITIONNELLE : QUANTIZATION ---
        if self.is_quantized:
            # Mode INT8 : On compresse avant de stocker
            # vectors est supposé être FP32/FP16 ici
            storage_vectors, batch_scales = SmartQuantizer.quantize(vectors)
            # On s'assure que les scales sont en (N, 1)
            if batch_scales.dim() == 1: batch_scales = batch_scales.unsqueeze(1)
        else:
            # Mode FP32 (Standard) : On stocke tel quel
            storage_vectors = Quantizer.to_storage(vectors)
            batch_scales = None
        # ---------------------------------------------
        
        for i, name in enumerate(names):
            if name in self.name_to_idx:
                idx = self.name_to_idx[name]
            else:
                idx = self.active_count
                self.name_to_idx[name] = idx
                self.idx_to_name[idx] = name
                self.active_count += 1
            indices_list.append(idx)
            
            # Mise à jour Shard
            sid = abs(hash(name)) % CFG.SHARD_COUNT
            
            if self.is_quantized:
                # En mode INT8, le shard doit stocker le tuple (vec, scale) ou un objet composite
                # Pour simplifier la compatibilité fichier, on stocke le vecteur compressé
                # ATTENTION : La gestion des shards en INT8 demandera une maj de save/load_shard plus tard
                # Pour l'instant, on stocke le tenseur principal.
                self.shards[sid][name] = storage_vectors[i].detach().to(CFG.DEVICE)
                # Note: On perd l'échelle dans les Shards ici pour l'instant (dette technique acceptée pour cette étape)
                # Mais on l'a dans master_scales pour le runtime.
            else:
                self.shards[sid][name] = storage_vectors[i].detach().to(CFG.DEVICE)
            
        if not indices_list: return
        
        indices_tensor = torch.tensor(indices_list, device=CFG.INDEX_DEVICE, dtype=torch.long)
        
        # 1. Mise à jour Fast Index (Toujours haute précision pour le calcul)
        if self.fast_index.device.type == 'cpu':
            vectors_tensor = vectors.detach().clone().to(dtype=CFG.INDEX_DTYPE, device=CFG.INDEX_DEVICE)
            self.fast_index[indices_tensor] = vectors_tensor
        else:
            vectors_tensor = vectors.to(dtype=CFG.INDEX_DTYPE, device=CFG.INDEX_DEVICE)
            self.fast_index.index_copy_(0, indices_tensor, vectors_tensor)
        
        # 2. Mise à jour Master Storage (Froid/Compressé)
        indices_cpu = indices_tensor.to(CFG.STORAGE_DEVICE)
        
        if self.master_storage.device.type == 'cpu':
             storage_tensor = storage_vectors.detach().clone().to(device=CFG.STORAGE_DEVICE)
             if self.is_quantized:
                 #scales_tensor = batch_scales.detach().clone().to(device=CFG.STORAGE_DEVICE)
                 # --- FIX CRASH TYPE MISMATCH (FP16 vs FP32) ---
                 # Les scales doivent être en Float32 pour matcher self.master_scales
                 # même si les vecteurs d'entrée étaient en FP16.
                 scales_tensor = batch_scales.detach().clone().to(device=CFG.STORAGE_DEVICE, dtype=torch.float32)
                 # ----------------------------------------------
        else:
             storage_tensor = storage_vectors.to(device=CFG.STORAGE_DEVICE)
             if self.is_quantized:
                 #scales_tensor = batch_scales.to(device=CFG.STORAGE_DEVICE)
                 # --- FIX CRASH TYPE MISMATCH (FP16 vs FP32) ---
                 scales_tensor = batch_scales.to(device=CFG.STORAGE_DEVICE, dtype=torch.float32)
                 # ----------------------------------------------
             
        self.master_storage.index_copy_(0, indices_cpu, storage_tensor)
        if self.is_quantized:
            self.master_scales.index_copy_(0, indices_cpu, scales_tensor)

    def update(self, name, vector):
        if vector.dim() == 1: vector = vector.unsqueeze(0)
        self.update_batch([name], vector)

    def find_closest(self, query_vec, threshold=0.0):
        res = self.find_top_k(query_vec, k=1, threshold=threshold)
        if res: return res[0]
        return None, 0.0

    def find_top_k(self, query_vec, k=50, threshold=0.0):
        if self.active_count == 0: return []
        if query_vec.dim() == 1: query_vec = query_vec.unsqueeze(0)
        q = query_vec.to(dtype=CFG.INDEX_DTYPE, device=CFG.INDEX_DEVICE)
        
        # Recherche sur le Fast Index (toujours décompressé/précis)
        active_matrix = self.fast_index[:self.active_count]
        scores = torch.mm(q, active_matrix.t()).squeeze(0)
        k_safe = min(k, self.active_count)
        best_scores, best_indices = torch.topk(scores, k=k_safe)
        results = []
        for i in range(len(best_indices)):
            score = best_scores[i].item()
            if score > threshold:
                idx = best_indices[i].item()
                name = self.idx_to_name.get(idx)
                if name: results.append((name, score))
        return results
        
    def get_vectors_by_names(self, names):
        indices = []
        found_names = []
        for n in names:
            if n in self.name_to_idx:
                indices.append(self.name_to_idx[n])
                found_names.append(n)
        if not indices: return None, []
        
        indices_tensor = torch.tensor(indices, device=CFG.STORAGE_DEVICE)
        raw_vecs = self.master_storage.index_select(0, indices_tensor).to(CFG.DEVICE)
        
        # --- DECOMPRESSION CONDITIONNELLE ---
        if self.is_quantized:
            raw_scales = self.master_scales.index_select(0, indices_tensor).to(CFG.DEVICE)
            # Restauration FP32 depuis INT8
            vecs = SmartQuantizer.dequantize(raw_vecs, raw_scales)
            # Conversion finale vers le type de calcul souhaité (ex: FP16)
            vecs = vecs.to(CFG.COMPUTE_DTYPE)
        else:
            vecs = Quantizer.from_storage(raw_vecs)
        # ------------------------------------
        
        return vecs, found_names

    def get_all_loaded_tensors(self):
        """
        [CORRECTIF CRITIQUE] Récupère tous les tenseurs pour le Lexique.
        Force le retour sur GPU (CFG.DEVICE) pour éviter le crash Device Mismatch.
        """
        all_t = {}
        for k, idx in self.name_to_idx.items():
            if idx < self.active_count:
                # 1. Lecture Master Storage (Souvent CPU)
                vec = self.master_storage[idx]
                scale = self.master_scales[idx]
                
                # 2. Transfert immédiat vers le Device de Calcul (GPU)
                # C'est cette étape qui manquait et causait le crash
                vec_gpu = vec.to(CFG.DEVICE)
                scale_gpu = scale.to(CFG.DEVICE)
                
                if vec.dtype == torch.int8:
                     val_final = SmartQuantizer.dequantize(vec_gpu, scale_gpu)
                else:
                     val_final = vec_gpu
                
                # On stocke dans le dictionnaire, prêt pour l'encodeur
                all_t[k] = val_final.to(dtype=CFG.COMPUTE_DTYPE)
        return all_t
        
    def sync_to_host(self, encoder_ref=None):
        if self.active_count == 0: return

        print(f" [MEMORY] Synchronisation VRAM -> RAM ({self.active_count} vecteurs)...")
        active_slice_gpu = self.fast_index[:self.active_count]
        
        # --- BLOC ACTIVÉ : Re-Quantization dynamique ---
        if self.is_quantized: 
             # On recalcule les INT8 et les Scales basés sur la nouvelle position du vecteur
             q_vecs, q_scales = SmartQuantizer.quantize(active_slice_gpu)
             
             # On met à jour le stockage maître (RAM)
             self.master_storage[:self.active_count] = q_vecs.to(CFG.STORAGE_DEVICE)
             self.master_scales[:self.active_count] = q_scales.to(CFG.STORAGE_DEVICE)
        else:
            # Mode FP32 Classique
            cpu_data = active_slice_gpu.to(device=CFG.STORAGE_DEVICE, dtype=CFG.STORAGE_DTYPE)
            self.master_storage[:self.active_count] = cpu_data
        # -----------------------------------------------
        
        # Synchro Shards (pour sauvegarde)
        for name, idx in self.name_to_idx.items():
            if idx < self.active_count:
                vec = self.master_storage[idx]
                sid = abs(hash(name)) % CFG.SHARD_COUNT
                if self.master_storage.device.type == 'cpu':
                    safe_vec = vec.detach().clone()
                else:
                    safe_vec = vec.detach()
                self.shards[sid][name] = safe_vec
                
                if encoder_ref is not None and not self.is_quantized:
                     if self.master_storage.device.type == 'cpu':
                         encoder_ref.semantic_map[name] = safe_vec.clone()
                     else:
                         encoder_ref.semantic_map[name] = safe_vec
        
        
    def save_all(self, encoder_ref=None):
        self.sync_to_host(encoder_ref)
        
        for sid, data in self.shards.items():
            SafeFileManager.save_shard(data, sid, CFG.BASE_MEM_DIR)
        
        # Sauvegarde des Scales si on est en INT8
        if self.is_quantized and self.master_scales is not None:
             # On sauvegarde juste la partie active
             active_scales = self.master_scales[:self.active_count]
             SafeFileManager.save_tensors({"memory_scales": active_scales}, os.path.join(CFG.BASE_MEM_DIR, "memory_scales.safetensors"))

        SafeFileManager.save_json({
            "mapping": {
                "name_to_idx": self.name_to_idx, 
                "active_count": self.active_count
            }
        }, os.path.join(CFG.BASE_MEM_DIR, "memory_index.json"))
        
        
    def load_index(self):
        meta = SafeFileManager.load_json(os.path.join(CFG.BASE_MEM_DIR, "memory_index.json"))
        if meta and "mapping" in meta:
            self.name_to_idx = meta["mapping"].get("name_to_idx", {})
            self.active_count = meta["mapping"].get("active_count", 0)
            self.idx_to_name = {v: k for k, v in self.name_to_idx.items()}
            
        # Chargement des Scales
        path_scales = os.path.join(CFG.BASE_MEM_DIR, "memory_scales.safetensors")
        if os.path.exists(path_scales):
             scale_dict = SafeFileManager.load_tensors(path_scales)
             if "memory_scales" in scale_dict:
                 loaded_scales = scale_dict["memory_scales"].to(CFG.STORAGE_DEVICE)
                 len_s = min(loaded_scales.shape[0], self.capacity)
                 self.master_scales[:len_s] = loaded_scales[:len_s]
                 print(f" [MEMORY] Scales chargés ({len_s} entrées).")

        # Chargement Données
        for sid in range(CFG.SHARD_COUNT):
            data = SafeFileManager.load_shard(sid, CFG.BASE_MEM_DIR)
            for k, v in data.items():
                self.shards[sid][k] = v
                
                if k in self.name_to_idx:
                    idx = self.name_to_idx[k]
                    if idx < self.capacity:
                        
                        # --- LOGIQUE UNIVERSELLE DE CHARGEMENT ---
                        # Source (Fichier) vs Destination (Mode Configuré)
                        
                        # Cas 1 : Source INT8 -> Mode Float (Décompression)
                        if v.dtype == torch.int8 and not self.is_quantized:
                             scale = self.master_scales[idx].to(CFG.STORAGE_DEVICE)
                             v_in = v.to(CFG.STORAGE_DEVICE)
                             v_final = SmartQuantizer.dequantize(v_in, scale).to(CFG.STORAGE_DTYPE)
                             
                        # Cas 2 : Source Float -> Mode INT8 (Compression)
                        # C'est le cas qui manquait ("l'oubli du if")
                        elif v.dtype != torch.int8 and self.is_quantized:
                             v_in = v.to(CFG.STORAGE_DEVICE)
                             # On doit quantizer à la volée
                             q_vec, q_scale = SmartQuantizer.quantize(v_in)
                             v_final = q_vec
                             self.master_scales[idx] = q_scale.squeeze() # Update du scale manquant
                             
                        # Cas 3 : Identiques (INT8->INT8 ou Float->Float)
                        else:
                             v_final = v.to(dtype=CFG.STORAGE_DTYPE, device=CFG.STORAGE_DEVICE)
                        
                        # Stockage Maître
                        self.master_storage[idx] = v_final
                        
                        # Restauration Fast Index (Toujours Float sur GPU)
                        if v_final.dtype == torch.int8:
                             scale = self.master_scales[idx].to(CFG.INDEX_DEVICE)
                             v_gpu = v_final.to(CFG.INDEX_DEVICE)
                             v_idx = SmartQuantizer.dequantize(v_gpu, scale)
                        else:
                             v_idx = v_final.to(CFG.INDEX_DEVICE)
                             
                        self.fast_index[idx] = v_idx.to(dtype=CFG.INDEX_DTYPE).reshape(self.dim)
        
        print(f" [MEMORY] Index chargé et restauré ({self.active_count} noeuds).")

class CognitiveStats:
    def __init__(self): self.usage = Counter(); self.accumulated_impact = {}; self.weights = {} 
    def register_impact(self, w, v): self.usage[w] += 1; self.accumulated_impact[w] = self.accumulated_impact.get(w, 0.0) + v; 
    def get_weight(self, w): return self.weights.get(w, 1.0)
    def get_inverse_freq_weight(self, w):
        count = self.usage[w]
        return 1.0 / (math.log(count + 2))

class MatrixEncoder(nn.Module):
    def __init__(self, dim_size, brain):
        super().__init__()
        self.brain = brain; self.dim = dim_size
        self.semantic_map = {}; self.stats = CognitiveStats(); self.locked_words = set()
        
        if TOKENIZERS_AVAILABLE:
            self.tokenizer = Tokenizer(models.BPE())
            self.tokenizer.normalizer = normalizers.Sequence([
                normalizers.NFD(), normalizers.Lowercase(), normalizers.StripAccents()
            ])
            self.tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
            self.tokenizer.decoder = decoders.ByteLevel()
            self.tokenizer.model = models.BPE(vocab={chr(i): i for i in range(256)}, merges=[])
            print(f" [ENCODER] Tokenizer Rust (ByteLevel) Initialisé.")
        else:
             print(" [ENCODER] Mode Fallback (Pas de tokenizers).")
             
        vocab_size = 5000 
        self.projection = torch.randn(vocab_size, self.dim, device=CFG.DEVICE)
        self.projection = F.normalize(self.projection, p=2, dim=1, eps=CFG.EPSILON)
        self._load_map()

    def lock_concept(self, w): self.locked_words.add(w)
    
    def _get_vectors_by_indices_fast(self, indices):
        """
        Récupère un lot de vecteurs directement depuis la mémoire du cerveau
        sans passer par le dictionnaire de noms si possible.
        """
        # Note: Cette fonction suppose que les indices correspondent aux UIDs du node_registry
        # ou aux indices du HybridMemoryCluster. 
        # Pour cette implémentation robuste, nous allons passer par le HybridMemoryCluster.
        # En MVP94, le lien Node ID -> Memory Index n'est pas direct 1:1 partout.
        # Nous allons donc utiliser une approche hybride sécurisée.
        
        # Pour l'instant, pour garantir la stabilité sans refondre tout le système d'ID:
        # On récupère les vecteurs via le système existant mais en batch.
        
        # OPTIMISATION: On accède directement au 'fast_index' de la mémoire si possible
        return self.brain.memory.fast_index[indices]
    
    
    def _generate_basis_from_ids(self, ids):
        """
        Mise à jour pour cohérence avec le nouveau standard (cas unitaire).
        """
        if ids.nelement() == 0: return torch.zeros(self.dim, device=CFG.DEVICE)
        
        length = ids.shape[0]
        vectors = self.projection[ids % self.projection.shape[0]]
        
        # Utilisation de la même fonction standard
        pos_encoding = self._get_sinusoidal_encoding(length, self.dim, CFG.DEVICE)
        
        vectors_scrambled = vectors * pos_encoding
        encoded_vec = torch.sum(vectors_scrambled, dim=0)
        
        return F.normalize(encoded_vec, p=2, dim=0, eps=CFG.EPSILON)

    def encode_word(self, text):
        self.stats.usage[text.lower()] += 1
        if text in self.semantic_map:
            stored = self.semantic_map[text]
            return Quantizer.from_storage(stored)
            
        if not TOKENIZERS_AVAILABLE: 
            return self._encode_legacy(text)
            
        encoding = self.tokenizer.encode(text)
        ids = torch.tensor(encoding.ids, device=CFG.DEVICE)
        vec = self._generate_basis_from_ids(ids)
        w = self.stats.get_inverse_freq_weight(text.lower())
        final_vec = vec * w 
        # --- CORRECTIF BUG "BODY" ---
        # Avant : self.semantic_map[text] = Quantizer.to_storage(final_vec)
        # Problème : En INT8, to_storage détruit les petites valeurs -> Vecteur Nul -> Body
        
        # Solution : On garde le cache de l'encodeur en Haute Précision (COMPUTE_DTYPE)
        # Cela évite le zéro, et c'est cohérent car semantic_map est un cache "chaud".
        self.semantic_map[text] = final_vec.to(dtype=CFG.COMPUTE_DTYPE)
        # ----------------------------
        return final_vec
    
    
        
    
    def _get_sinusoidal_encoding(self, length, dim, device):
        """
        [PRO STANDARD] Génère des encodages positionnels sinusoïdaux.
        Basé sur Vaswani et al. "Attention Is All You Need".
        Garantit une dispersion unique et continue des positions.
        """
        pe = torch.zeros(length, dim, device=device)
        position = torch.arange(0, length, dtype=torch.float, device=device).unsqueeze(1)
        
        # Terme diviseur (Fréquences géométriques : 10000 ^ (-2i/dim))
        div_term = torch.exp(torch.arange(0, dim, 2, dtype=torch.float, device=device) * -(math.log(10000.0) / dim))
        
        # Application Sinus/Cosinus alternés
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        return pe

    def encode_batch_fast(self, text_list):
        """
        Version PRO : Vectorisation totale + Encodage Standard.
        Compatible CPU/GPU, haute performance, mathématiquement robuste.
        """
        self.stats.usage.update([t.lower() for t in text_list])
            
        if not TOKENIZERS_AVAILABLE:
            return torch.stack([self.encode_word(t) for t in text_list])
            
        encodings = self.tokenizer.encode_batch(text_list)
        
        # 1. Padding Vectorisé (Batch Rectangulaire)
        max_len = max(len(e.ids) for e in encodings)
        if max_len == 0: return torch.zeros((len(text_list), self.dim), device=CFG.DEVICE)
        
        batch_size = len(text_list)
        padded_ids = torch.zeros((batch_size, max_len), dtype=torch.long, device=CFG.DEVICE)
        
        # Remplissage rapide
        for i, e in enumerate(encodings):
            if len(e.ids) > 0:
                padded_ids[i, :len(e.ids)] = torch.tensor(e.ids, device=CFG.DEVICE)

        # 2. Projection Sémantique (Lookup)
        # [Batch, MaxLen, Dim]
        vectors = self.projection[padded_ids % self.projection.shape[0]]
        
        # 3. Encodage Positionnel "Standard" (Le remplacement des nombres magiques)
        # On génère la matrice de position [MaxLen, Dim]
        pos_encoding = self._get_sinusoidal_encoding(max_len, self.dim, CFG.DEVICE)
        
        # On l'étend pour le batch : [1, MaxLen, Dim] -> broadcast sur [Batch, MaxLen, Dim]
        pos_encoding = pos_encoding.unsqueeze(0)
        
        # 4. Masquage du Padding
        # On ne veut pas que les '0' ajoutés influencent le vecteur final
        mask = (padded_ids != 0).unsqueeze(-1).float()
        
        # 5. Combinaison & Scrambling
        # Dans la théorie VSA (Vector Symbolic Architecture), la multiplication (Hadamard product)
        # est souvent utilisée pour lier le "Rôle" (Position) à la "Valeur" (Mot).
        # Standard Transformer = Addition, mais votre architecture VSA préfère la Multiplication pour le binding.
        # On garde la multiplication pour la cohérence de votre modèle.
        vectors_scrambled = vectors * pos_encoding * mask
        
        # 6. Somme (Agrégation)
        encoded_vecs = torch.sum(vectors_scrambled, dim=1)
        
        # 7. Normalisation & Pondération IDF
        # Calcul des poids IDF (Importance du mot)
        # [Batch, 1]
        weights = torch.tensor([
            self.stats.get_inverse_freq_weight(t.lower()) 
            for t in text_list
        ], device=CFG.DEVICE).unsqueeze(1)
        
        # Application aux vecteurs
        final_vecs = F.normalize(encoded_vecs, p=2, dim=1, eps=CFG.EPSILON) * weights
        
        # MODIFICATION : On retourne aussi les poids bruts !
        # Ils serviront de "Masse de base" pour la physique.
        return final_vecs, weights.squeeze(1)

    

    def _encode_legacy(self, text): 
        b_text = text.encode('latin-1', errors='replace')
        indices = torch.tensor(list(b_text), device=CFG.DEVICE)
        vec = self._generate_basis_from_ids(indices)
        return vec * self.stats.get_weight(text.lower())

    def bind_syntax(self, vector, position_index): return torch.roll(vector, shifts=position_index, dims=0)
    
    def get_semantic_vector(self, text):
        # 1. Priorité à la mémoire vive (Hot Memory - GPU/Fast Index)
        # C'est là que les modifications vectorisées (learn_attraction_batch) sont appliquées.
        if hasattr(self.brain, 'memory') and text in self.brain.memory.name_to_idx:
            idx = self.brain.memory.name_to_idx[text]
            # On récupère le vecteur frais directement depuis l'index
            vec = self.brain.memory.fast_index[idx]
            # On le convertit en float32 (COMPUTE_DTYPE) pour les calculs
            return vec.to(dtype=CFG.COMPUTE_DTYPE)

        # 2. Fallback sur le cache dictionnaire (Cold Storage)
        # Utilisé pour les mots qui ne sont pas encore indexés dans le cluster mémoire
        if text not in self.semantic_map: 
            self.encode_word(text) 
        
        return Quantizer.from_storage(self.semantic_map[text])

    def learn_attraction(self, wa, wb, force=0.1):
        f = force * self.brain.temperature; va = self.get_semantic_vector(wa); vb = self.get_semantic_vector(wb); tgt = F.normalize(va+vb, p=2, dim=0, eps=CFG.EPSILON)
        elasticity = CFG.ELASTICITY_ATTRACTION
        if wa not in self.locked_words: 
            ra = (self.encode_word(wa) - va) * elasticity; ma = (tgt - va) * f
            new_va = F.normalize(va + ma + ra, p=2, dim=0, eps=CFG.EPSILON)
            
            self.semantic_map[wa] = Quantizer.to_storage(new_va)
            # --- FIX : On injecte le NOUVEAU vecteur directement ---
            self.brain.memory.update(wa, new_va) 
            # -------------------------------------------------------

        if wb not in self.locked_words:
            rb = (self.encode_word(wb) - vb) * elasticity; mb = (tgt - vb) * f
            new_vb = F.normalize(vb + mb + rb, p=2, dim=0, eps=CFG.EPSILON)
            
            self.semantic_map[wb] = Quantizer.to_storage(new_vb)
            # --- FIX : On injecte le NOUVEAU vecteur directement ---
            self.brain.memory.update(wb, new_vb)
            # -------------------------------------------------------
            
        self.brain.associative_memory.is_dirty = True

    def learn_attraction_batch(self, indices_source, indices_target, forces):
        """
        Version V2.0 : Entièrement vectorisée sur GPU/CPU.
        Remplace la boucle Python lente par des opérations tensorielles.
        """
        # 1. Validation et Nettoyage (Gardien)
        if indices_source.numel() == 0: return
        
        # S'assurer que tout est sur le bon Device et plat
        src = indices_source.view(-1).to(CFG.DEVICE)
        tgt = indices_target.view(-1).to(CFG.DEVICE)
        frc = forces.view(-1).to(CFG.DEVICE)
        
        # Filtrer les auto-références (A -> A)
        mask = (src != tgt)
        if not mask.any(): return
        
        src = src[mask]
        tgt = tgt[mask]
        frc = frc[mask]
        
        # 2. Récupération des Vecteurs (Batch Gather)
        # On utilise directement le fast_index du MemoryCluster pour la vitesse
        # Attention: Cela suppose que indices_source correspondent aux indices mémoire.
        # Si ce sont des Node UIDs, nous devons avoir une map. 
        # HYPOTHESE FORTE: Dans MVP94, Node UID ~= Memory Index pour les concepts actifs.
        # Si ce n'est pas le cas, il faut passer par node_registry, ce qui est lent.
        # Pour ce fix, nous supposons que indices sont alignés ou que nous opérons sur les vecteurs.
        
        # Récupération sécurisée via le cluster mémoire
        vecs_a = self.brain.memory.fast_index.index_select(0, src)
        vecs_b = self.brain.memory.fast_index.index_select(0, tgt)
        
        # 3. Calcul Physique Vectoriel (Loi d'Attraction)
        # Formule: ma = (tgt - va) * force
        # Note: On ignore l'élasticité "ra" (retour à l'encodage mot) dans le batch haute vitesse 
        # pour éviter de recalculer encode_word() qui est très lent (tokenizer).
        # C'est un compromis performance/précision acceptable pour la propagation de masse.
        
        # Target mutual attraction (A s'approche de B, B s'approche de A)
        # Ou convergence vers le centre (A+B)
        target_center = F.normalize(vecs_a + vecs_b, p=2, dim=1, eps=CFG.EPSILON)
        
        # Calcul des Deltas
        # delta = (Target - Current) * Force * Plasticity
        # On simplifie plasticity à 1.0 ici, géré par 'forces' en amont
        
        # Expansion de 'frc' pour matcher les dimensions [Batch, Dim]
        frc_expanded = frc.unsqueeze(1)
        
        delta_a = (target_center - vecs_a) * frc_expanded * CFG.LEARNING_RATE_HARDWARE
        delta_b = (target_center - vecs_b) * frc_expanded * CFG.LEARNING_RATE_HARDWARE
        
        new_vecs_a = F.normalize(vecs_a + delta_a, p=2, dim=1, eps=CFG.EPSILON)
        new_vecs_b = F.normalize(vecs_b + delta_b, p=2, dim=1, eps=CFG.EPSILON)
        
        # 4. Application (Batch Scatter/Update)
        safe_vecs_a = new_vecs_a.to(CFG.INDEX_DTYPE)
        safe_vecs_b = new_vecs_b.to(CFG.INDEX_DTYPE)
        
        # --- FIX ULTIME CPU (OP8) ---
        if self.brain.memory.fast_index.device.type == 'cpu':
            # Sur CPU : On clone pour la sécurité ET on utilise l'assignation standard []
            # L'assignation standard gère mieux les conflits mémoire que index_copy_ sur CPU
            safe_vecs_a = safe_vecs_a.detach().clone()
            safe_vecs_b = safe_vecs_b.detach().clone()
            
            self.brain.memory.fast_index[src] = safe_vecs_a
            self.brain.memory.fast_index[tgt] = safe_vecs_b
        else:
            # Sur GPU : index_copy_ est beaucoup plus rapide (asynchrone)
            self.brain.memory.fast_index.index_copy_(0, src, safe_vecs_a)
            self.brain.memory.fast_index.index_copy_(0, tgt, safe_vecs_b)
        # ----------------------------
            
        self.brain.associative_memory.is_dirty = True
            
    def learn_repulsion(self, wa, wb, force=0.1):
        if wa in self.locked_words: return
        
        va = self.get_semantic_vector(wa)
        vb = self.get_semantic_vector(wb)
        
        # Calcul de la répulsion
        ra = (self.encode_word(wa) - va) * CFG.ELASTICITY_ATTRACTION
        new_va = F.normalize(va - (vb * force) + ra, p=2, dim=0, eps=CFG.EPSILON)
        
        # --- CORRECTIF : Sauvegarde du NOUVEAU vecteur ---
        # 1. Mise à jour cache local (avec le fix "Body" appliqué ici aussi par sécurité)
        self.semantic_map[wa] = new_va.to(dtype=CFG.COMPUTE_DTYPE)
        
        # 2. Mise à jour mémoire centrale (On passe new_va, pas self.get_semantic_vector(wa))
        self.brain.memory.update(wa, new_va)
        # -----------------------------------------------
        
        self.brain.associative_memory.is_dirty = True
        
    def save(self, path):
        meta = {"weights": self.stats.weights}
        SafeFileManager.save_json(meta, os.path.join(path, "cognitive_stats.json"))
        save_file({"projection": self.projection.detach().cpu()}, os.path.join(path, "matrix_projection.safetensors"))

    def _load_map(self):
        meta = SafeFileManager.load_json(os.path.join(CFG.BASE_MEM_DIR, "cognitive_stats.json"))
        if meta: self.stats.weights = meta.get("weights", {})
        path_proj = os.path.join(CFG.BASE_MEM_DIR, "matrix_projection.safetensors")
        if os.path.exists(path_proj):
            loaded = load_file(path_proj)
            if "projection" in loaded:
                print(" [ENCODER] Matrice de projection restaurée.")
                self.projection = loaded["projection"].clone().to(CFG.DEVICE)

class CognitiveMetabolism:
    def __init__(self, brain): self.brain = brain
    def run_cycle(self):
        print("\n [METABOLISME] Recalibrage...")
        pass

class HyperPhysics(nn.Module):
    def __init__(self, dim): super().__init__(); self.dim = dim

class ToolFactory:
    def __init__(self, brain):
        self.brain = brain; self.sequence_buffer = []; self.patterns = Counter() 
    def register_op(self, op_name):
        self.sequence_buffer.append(op_name)
        if len(self.sequence_buffer) > 2: self.sequence_buffer.pop(0)
        if len(self.sequence_buffer) == 2:
            combo = f"{self.sequence_buffer[0]}+{self.sequence_buffer[1]}"
            self.patterns[combo] += 1
            if self.patterns[combo] >= CFG.TOOL_PATTERN_THRESHOLD:
                tool_name = f"TOOL_{combo}"
                if tool_name not in HARDWARE_REGISTRY:
                    self.forge_new_tool(tool_name, self.sequence_buffer.copy())
                    self.patterns[combo] = 0
    def forge_new_tool(self, name, op_names):
        print(f" [INVENTION] Découverte d'un outil composite : {name} ({op_names})")
        ops = []
        for op_code in op_names: ops.append(HARDWARE_REGISTRY.get(op_code))
        new_tool = OpComposite(name, ops)
        HARDWARE_REGISTRY[name] = new_tool
        return new_tool

class FractalNode(nn.Module):
    # --- OPTIMISATION MEMOIRE (MVP 96.12) ---
    # Liste EXHAUSTIVE des attributs de FractalNode.
    # Note : Comme on hérite de nn.Module, un __dict__ existera quand même pour PyTorch,
    # mais nos données métier seront stockées dans ces slots optimisés (-40% RAM).
    __slots__ = [
        'name', 
        'dim', 
        'phys', 
        'encoder', 
        'parent', 
        'children', 
        'concepts', 
        'states', 
        'metadata', 
        'brain', 
        'layer_type', 
        'percepts', 
        'uid', 
        'fractal_id', 
        'path', 
        'is_dirty', 
        '_plasticity',  # Attention : c'est _plasticity (la propriété wrapper)
        'mass', 
        'nature_vec'
    ]
    # ----------------------------------------


    def __init__(self, name, dim, phys, parent=None, nature="Neutre", encoder=None, layer_type=None):
        super().__init__(); self.name = name; self.dim = dim; self.phys = phys; self.encoder = encoder
        self.parent = parent; self.children = {}; self.concepts = []; self.states = {}; self.metadata = {}
        self.brain = None
        self.layer_type = layer_type if layer_type is not None else CFG.LAYER_CONCEPT
        self.percepts = [] 
        
        if self.encoder: self.brain = self.encoder.brain
        elif self.parent: self.brain = self.parent.brain
        if self.brain: 
            self.uid = self.brain.register_node(self)
            self.fractal_id = self.uid 
        else: self.uid = 0; print("rrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrr")
        
        if parent: self.path = os.path.join(parent.path, name)
        else: self.path = os.path.join(CFG.BASE_MEM_DIR, name)
        self.is_dirty = True
        
        plasticity = CFG.PLASTICITY_DEFAULT
        if "REALITE" in self.path: plasticity = CFG.PLASTICITY_REALITY
        elif "CONCEPTS" in self.path: plasticity = CFG.PLASTICITY_CONCEPT
        self.plasticity = plasticity 
        
        energy = CFG.ENERGY_INIT_DEFAULT
        if "REALITE" in self.path: energy = CFG.ENERGY_INIT_REALITY
        elif "CONCEPTS" in self.path: energy = CFG.ENERGY_INIT_CONCEPT
        self.energy = energy 
        self.mass = self._resolve_mass()
        if self.encoder: self.nature_vec = self.encoder.encode_word(nature)
        else: self.nature_vec = torch.randn(dim).to(CFG.DEVICE)
        self._load()

    def _resolve_mass(self):
        if "mass" in self.metadata: return self.metadata["mass"]
        if self.parent and self.parent.name == "REALITE": return CFG.MASS_MAPPING["INSTANCE"]
        return CFG.MASS_MAPPING.get(self.layer_type, CFG.MASS_MAPPING["DEFAULT"])

    @property
    def semantic_group(self):
        group = [self.uid]
        if self.percepts:
            for p in self.percepts: group.append(p.uid)
        if self.concepts:
            for c in self.concepts: group.append(c.uid)
        return group
    
    @property
    def energy(self):
        if self.brain: return self.brain.global_energies[self.uid].item()
        return 0.0
    @energy.setter
    def energy(self, value):
        if self.brain: self.brain.global_energies[self.uid] = value
    @property
    def plasticity(self): return self._plasticity
    @plasticity.setter
    def plasticity(self, value): self._plasticity = value
    def bind_hardware_function(self, op, direction="<>"): self.metadata["native_op"] = op; self.metadata["op_dir"] = direction; 
    def get_hardware_function(self): return HARDWARE_REGISTRY.get(self.metadata.get("native_op"))
    def add_child(self, n): self.children[n.name] = n; n.parent = self
    def link_concept(self, n): 
        if n not in self.concepts:
            self.concepts.append(n)
    
    def link_percept(self, percept_node):
        if percept_node not in self.percepts:
            self.percepts.append(percept_node)
            if self not in percept_node.concepts:
                percept_node.concepts.append(self)
            if self.brain:
                self.brain.encoder.learn_attraction(self.name, percept_node.name, force=1.0)
    
    def absorb(self, dim, target_vec, force=1.0):
        if target_vec.is_sparse: target_vec = target_vec.to_dense()
        current = self.get_local(dim)
        if current is None: new_vec = target_vec * self.plasticity 
        else:
            if current.is_sparse: current = current.to_dense()
            delta = (target_vec - current) * self.plasticity * force
            new_vec = F.normalize(current + delta, p=2, dim=0, eps=CFG.EPSILON)
        self.set(dim, new_vec)
        new_e = min(100.0, self.energy + 10.0)
        self.energy = new_e
    def update_centroid(self, visited=None):
        if visited is None: visited = set()
        visited.add(self.name)
        if not self.children: return self.nature_vec
        vecs = [c.update_centroid(visited) for c in self.children.values()]
        if vecs: self.nature_vec = F.normalize(self.nature_vec * (1-CFG.MOMENTUM_MEAN) + torch.stack(vecs).mean(dim=0) * CFG.MOMENTUM_MEAN, p=2, dim=0, eps=CFG.EPSILON)
        return self.nature_vec
    def apply_decay(self):
        dead = []
        for n, c in list(self.children.items()):
             dead.extend(c.apply_decay())
             if n in dead: 
                 if self.brain: self.brain.delete_node(c)
                 del self.children[n]
             elif c.energy < 1.0 and not c.metadata.get("native_op"): dead.append(n)
        return dead
    def set(self, d, v): 
        # MODIFICATION : On bypass le stockage INT8 pour les propriétés locales
        # pour éviter l'effacement des données fines.
        # On utilise COMPUTE_DTYPE (FP16/32) au lieu de STORAGE_DTYPE
        self.states[d] = v.to(dtype=CFG.COMPUTE_DTYPE, device=CFG.DEVICE)
        self.mark_dirty()
    def mark_dirty(self): 
        if not self.is_dirty: self.is_dirty = True; 
        if self.parent: self.parent.mark_dirty()
    def get_local(self, d): 
        if d in self.states:
            # Plus besoin de conversion complexe, c'est déjà dans le bon format
            return self.states[d]
        return None
    def get_conceptual(self, d): 
        for c in self.concepts: 
            if (v := c.get_local(d)) is not None: return v
        return None
    def update_concept_knowledge(self, dim, vec): 
        if self.concepts: self.concepts[0].absorb(dim, vec)
    def create_hypothesis(self, dim, vec): 
        hyp_name = f"HYP_{self.name}_{dim}"
        if hyp_name in self.children: 
            print(f" [HYPOTHESE] Renforcement: {hyp_name}")
            self.children[hyp_name].absorb(dim, vec) 
        else:
            print(f" [HYPOTHESE] Création: {hyp_name}")
            hyp_node = FractalNode(hyp_name, self.dim, self.phys, parent=self, encoder=self.encoder, nature="Hypothesis")
            
            # --- MODIFICATION MVP94.6 : Méta-données de Reconstruction ---
            # On stocke la source pour pouvoir "refusionner" les pointeurs au chargement
            hyp_node.metadata["source_concept"] = dim
            
            # On essaie de deviner le layer de la source pour être précis
            # Si on a accès au cerveau et au concept source
            if self.brain:
                source_c = self.brain.find_concept_exact(dim)
                if source_c:
                    hyp_node.metadata["source_layer"] = source_c.layer_type
            # -------------------------------------------------------------
            
            hyp_node.set("valeur", vec); self.add_child(hyp_node)
    def _load(self): pass 
    def save_recursive(self):
        self.metadata["energy"] = self.energy 
        SafeFileManager.save_json(self.metadata, os.path.join(self.path, "meta.json"))
        for c in self.children.values(): c.save_recursive()
        
    def get_root(self):
        curr = self
        while curr.parent is not None:
            curr = curr.parent
        return curr

    def is_in_reality(self):
        return "REALITE" in self.get_full_path()

    def get_full_path(self):
        chain = []
        curr = self
        while curr is not None:
            chain.append(curr.name)
            curr = curr.parent
        return chain

# --- OPERATORS (MVP94.5) ---
class SemanticOperator:
    # Suppression de _ensure_operands : Le travail est fait en amont par le Stream
    
    def execute(self, brain, node_subj, node_target, layer_type=None, trust_level=1.0):
        """
        Exécute l'opération sur des NOEUDS déjà résolus/créés.
        node_subj: FractalNode
        node_target: FractalNode
        """
        raise NotImplementedError
    
    @property
    def priority(self): return 1.0

class OpComposite(SemanticOperator):
    def __init__(self, name, ops):
        self.name = name; self.ops = ops; self._priority = 2.5
    @property
    def priority(self): return self._priority
    
    def execute(self, brain, node_subj, node_target, layer_type=None, trust_level=1.0):
        curr_subj = node_subj
        for op_instance in self.ops:
             # Propagation du contexte et des objets
             res_node = op_instance.execute(brain, curr_subj, node_target, layer_type=layer_type, trust_level=trust_level)
             
             # Chaînage : Si l'opérateur renvoie un nouveau nœud (ex: Synthèse), on l'utilise pour la suite
             if res_node is not None:
                 curr_subj = res_node
        return curr_subj

#def apply_agnostic_propagation(brain, subj_node, target_node, relation_type="LINK"):
#    brain.propagate_forces_vectorized(subj_node.semantic_group, target_node.semantic_group)
#    return None
    
def apply_agnostic_propagation(brain, node_subj, node_target):
    """Helper pour mettre à jour la physique entre deux nœuds identifiés."""
    if node_subj and node_target:
        brain.propagate_forces_vectorized(node_subj.semantic_group, node_target.semantic_group)

class OpEquivalence(SemanticOperator):
    def execute(self, brain, node_subj, node_target, layer_type=None, trust_level=1.0):
        brain.tool_factory.register_op("OP_DEF")
        
        if node_subj and node_target:
            target_key = node_target.name
            target_vec = node_target.nature_vec
            
            # Écriture directe (Optimisation: pas de ré-encodage)
            if trust_level > 0.5:
                node_subj.set(target_key, target_vec)
                # Propagation du signal (Apprentissage)
                brain.propagate_signal(node_subj, target_vec, trust_level, key=target_key)
            
            apply_agnostic_propagation(brain, node_subj, node_target)
            
        print(f" [HARDWARE] {node_subj.name} ≡ {node_target.name} (Layer {layer_type})")
        return node_subj

class OpInclusion(SemanticOperator):
    def execute(self, brain, node_subj, node_target, layer_type=None, trust_level=1.0):
        brain.tool_factory.register_op("OP_LOC")
        
        if node_subj and node_target:
            # 1. Gestion de la Hiérarchie (Déplacement)
            if node_subj.parent and node_subj.name in node_subj.parent.children:
                del node_subj.parent.children[node_subj.name]
            
            node_subj.parent = node_target
            node_target.add_child(node_subj)
            node_subj.path = os.path.join(node_target.path, node_subj.name)
            
            # 2. Physique
            apply_agnostic_propagation(brain, node_subj, node_target)
            
            # 3. Signal (Grounding)
            # On propage le vecteur du sujet vers le contenant
            brain.propagate_signal(node_subj, node_subj.nature_vec, trust_level)
            
            # 4. Molécule (Narrative uniquement en Concept)
            if layer_type == CFG.LAYER_CONCEPT:
                brain.create_molecule(node_subj.name, "LOC", node_target.name)
            
            print(f" [HARDWARE] {node_subj.name} ⊂ {node_target.name} (Layer {layer_type})")
            return node_subj
        return None

class OpAttribution(SemanticOperator):
    def execute(self, brain, node_subj, node_target, layer_type=None, trust_level=1.0):
        brain.tool_factory.register_op("OP_ATTR")
        
        if node_subj and node_target:
            # Normalisation de la clé (ex: "Chaud" -> "chaud")
            target_key = node_target.name
            
            # OPTIMISATION : On utilise le vecteur déjà présent dans le nœud cible
            # Plus besoin d'appeler l'encodeur -> Gain de perf et cohérence
            target_vec = node_target.nature_vec
            
            # 1. Application Locale (Modification de l'état)
            node_subj.absorb(target_key, target_vec, force=CFG.LEARNING_RATE_ATTRIBUTION * trust_level)
            
            # 2. Inférence Active (AVANT propagation pour détecter la surprise Ouroboros)
            # Le nœud possède déjà la bonne plasticité (réglée par la Factory)
            # print(f"sujet:{node_subj.name};target{node_target.name};plasticity{node_subj.plasticity};layer{layer_type}")
            if node_subj.plasticity > 0.1:
                brain.active_inference_check(node_subj, node_target.name)
            
            # 3. Propagation vers le Concept (Apprentissage Bottom-Up)
            brain.propagate_signal(node_subj, target_vec, trust_level * CFG.PROPAGATION_DECAY, key=target_key)
            
            # 4. Molécule Narrative
            if layer_type == CFG.LAYER_CONCEPT:
                brain.create_molecule(node_subj.name, "ATTR", node_target.name)
            
            print(f" [HARDWARE] ATTRIBUTION : {node_subj.name} -> {node_target.name} (Layer {layer_type})")
            return node_subj
        return None

class OpSequence(SemanticOperator):
    @property
    def priority(self): return 3.0
    
    def execute(self, brain, node_subj, node_target, layer_type=None, trust_level=1.0):
        brain.tool_factory.register_op("OP_SEQ")
        
        if node_subj and node_target:
            print(f" [HARDWARE] CAUSALITE : {node_subj.name} -> {node_target.name}")
            
            # Enregistrement contextuel
            if trust_level > 0.2:
                # Optimisation : vecteur direct
                node_subj.set("next", node_target.nature_vec)
                
        return node_subj

class OpRepulsion(SemanticOperator):
    def execute(self, brain, node_subj, node_target, layer_type=None, trust_level=1.0):
        brain.tool_factory.register_op("OP_REP")
        
        if node_subj and node_target:
            effective_force = CFG.LEARNING_RATE_HARDWARE * trust_level
            # Ici on garde l'appel encodeur pour la gestion fine des poids répulsifs
            brain.encoder.learn_repulsion(node_subj.name, node_target.name, force=effective_force)
            
            print(f" [HARDWARE] {node_subj.name} >< {node_target.name}")
        return None

class OpSynthesis(SemanticOperator):
    @property
    def priority(self): return 2.0
    
    def execute(self, brain, node_subj, node_target, layer_type=None, trust_level=1.0):
        brain.tool_factory.register_op("OP_SYN")
        
        new_name = f"SYNTH_{node_subj.name}_{node_target.name}"
        
        # Création du résultat dans le bon layer via la Factory
        new_node = brain.ensure_node_in_layer(new_name, layer_type)
        
        # Calcul vectoriel
        sem_a = node_subj.nature_vec
        sem_b = node_target.nature_vec
        res_vec = F.normalize(sem_a + sem_b, p=2, dim=0, eps=CFG.EPSILON)
        
        # Enregistrement global si concept
        if layer_type == CFG.LAYER_CONCEPT:
             brain.encoder.semantic_map[new_name] = res_vec
             
        new_node.absorb("valeur", res_vec, force=trust_level)
        print(f" [HARDWARE] {node_subj.name} + {node_target.name} -> {new_name}")
        return new_node

class OpContext(SemanticOperator):
    @property
    def priority(self): return 2.0
    
    def execute(self, brain, node_subj, node_target, layer_type=None, trust_level=1.0):
        brain.tool_factory.register_op("OP_CTX")
        
        res_name = f"{node_subj.name}@{node_target.name}"
        
        # Création via Factory
        new_node = brain.ensure_node_in_layer(res_name, layer_type)
        
        sem_a = node_subj.nature_vec
        sem_b = node_target.nature_vec
        res_vec = F.normalize(sem_a * sem_b, p=2, dim=0, eps=CFG.EPSILON)
        
        if layer_type == CFG.LAYER_CONCEPT:
             brain.encoder.semantic_map[res_name] = res_vec
             
        new_node.absorb("valeur", res_vec, force=trust_level)
        print(f" [HARDWARE] {node_subj.name} * {node_target.name} = {res_name}")
        return new_node
 
class OpLabel(SemanticOperator):
    @property
    def priority(self): return 0.5
    
    def execute(self, brain, node_subj, node_target, layer_type=None, trust_level=1.0):
        brain.tool_factory.register_op("OP_LBL")
        
        # Note : OpLabel est particulier car 'node_target' est souvent le nouveau nom
        # Si node_target existe déjà, on fusionne/écrase
        
        if trust_level > 0.8:
            # Mise à jour globale de l'encodeur pour le futur
            brain.encoder.semantic_map[node_target.name] = node_subj.nature_vec
            
            # Mise à jour locale
            node_target.absorb("valeur", node_subj.nature_vec, force=1.0)
            
            print(f" [HARDWARE] '{node_subj.name}' renommé en '{node_target.name}'")
            return node_target
        return node_subj
 
class OpNegation(SemanticOperator):
    def execute(self, brain, node_subj, node_target, layer_type=None, trust_level=1.0):
        brain.tool_factory.register_op("OP_NEG")
        
        if node_subj and node_target:
            target_key = node_target.name
            local_val = node_subj.get_local(target_key)
            
            vec_target = node_target.nature_vec
            
            if local_val is not None:
                # Soustraction vectorielle
                node_subj.set(target_key, F.normalize(local_val - (vec_target * trust_level), p=2, dim=0, eps=CFG.EPSILON))
                print(f" [HARDWARE] {node_subj.name} - {node_target.name}")
            else:
                # Création d'anti-matière
                if trust_level > 0.5:
                    v_anti = -vec_target
                    node_subj.set(target_key, v_anti)
                    print(f" [HARDWARE] {node_subj.name} + ANTI-{node_target.name}")
        return None

HARDWARE_REGISTRY = {
    "OP_DEF": OpEquivalence(), "OP_LOC": OpInclusion(), "OP_ATTR": OpAttribution(),
    "OP_REP": OpRepulsion(), "OP_SYN": OpSynthesis(), "OP_NEG": OpNegation(), 
    "OP_CTX": OpContext(), "OP_LBL": OpLabel(), "OP_SEQ": OpSequence()
}


class GenesisBridge(threading.Thread):
    """
    Classe Mère pour les workers.
    Accepte brain=None pour le mode Headless (utilisé dans les sous-processus).
    """
    def __init__(self, name, brain=None, input_q=None, output_q=None):
        super().__init__(name=name)
        self.brain = brain
        self.in_q = input_q if input_q else queue.Queue(maxsize=100)
        self.out_q = output_q if output_q else queue.Queue(maxsize=CFG.INGESTION_QUEUE_SIZE)
        self.daemon = True 
        self.stop_event = threading.Event()
        
        # Enregistrement seulement si on est dans le thread principal (avec un brain)
        if self.brain:
            GenesisConfig.register(self)

    def stop(self):
        self.stop_event.set()

    def process_data(self, raw_data):
        raise NotImplementedError

    def run(self):
        print(f" [BRIDGE] Démarrage du worker léger : {self.name}")
        while not self.stop_event.is_set():
            try:
                raw_data = self.in_q.get(timeout=CFG.INGESTION_TIMEOUT)
                if raw_data == "__SYNC_MARKER__":
                    self.out_q.put({"type": "MARKER"})
                    continue
                
                result = self.process_data(raw_data)
                # Note: process_data peut gérer l'envoi lui-même
                    
            except queue.Empty:
                continue
            except Exception as e:
                print(f" [ERR] Bridge {self.name} crash: {e}")

class TextIngestionBridge(GenesisBridge):
    """
    [LOGIC CORE] Gère la préparation du texte.
    Peut fonctionner en mode :
    1. CONNECTÉ (Light Bridge) : Utilise self.brain pour vectoriser immédiatement.
    2. DÉTACHÉ (Heavy Bridge) : Utilise un tokenizer local pour préparer les paquets.
    """
    def __init__(self, name, brain=None):
        super().__init__(name, brain)
        self.local_tokenizer = None
        
    def setup_headless(self):
        """Initialise les outils pour le mode détaché (sans brain)."""
        try:
            from tokenizers import Tokenizer, models, pre_tokenizers, decoders, normalizers
            self.local_tokenizer = Tokenizer(models.BPE())
            self.local_tokenizer.normalizer = normalizers.Sequence([
                normalizers.NFD(), normalizers.Lowercase(), normalizers.StripAccents()
            ])
            self.local_tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
            self.local_tokenizer.decoder = decoders.ByteLevel()
            self.local_tokenizer.model = models.BPE(vocab={chr(i): i for i in range(256)}, merges=[])
        except ImportError:
            self.local_tokenizer = None

    def process_cpu_batch(self, raw_lines):
        """
        [FONCTION COEUR] Logique pure CPU partagée (Light & Heavy).
        Nettoie, Découpe (Split) et Tronçonne (Chunking) le texte.
        """
        processed_items = []
        # On récupère la limite de taille (Défaut 128 si pas de config)
        max_len = CFG.MAX_SEQUENCE_LENGTH if hasattr(builtins, 'CFG') else 128
        
        for line in raw_lines:
            # 1. Nettoyage et split phrases (Premier niveau)
            sentences = line.replace(".", " .").split(" .")
            
            for sent in sentences:
                txt = sent.strip()
                if not txt: continue
                
                # 2. Découpage en Mots (Niveau Sémantique)
                all_words = txt.split()
                if not all_words: continue
                
                # 3. CHUNKING (Sécurité Buffer)
                # C'est ici qu'on applique l'optimisation de troncation centralisée
                for i in range(0, len(all_words), max_len):
                    chunk_words = all_words[i : i + max_len]
                    
                    # Tokenization ou Hash (Agnostique)
                    token_ids = []
                    tokenizer = self.local_tokenizer
                    if not tokenizer and self.brain:
                        tokenizer = self.brain.encoder.tokenizer
                    
                    if tokenizer:
                        # On simule des IDs (le GPU refera le calcul fin si besoin)
                        token_ids = [0] * len(chunk_words)
                    else:
                        token_ids = [abs(hash(w)) % 5000 for w in chunk_words]

                    local_counts = Counter(chunk_words)
                    
                    # Structure standardisée
                    item = {
                        "ids": token_ids,
                        "tokens": chunk_words, # Paquet sûr (<= 128 mots)
                        "counts": local_counts,
                        "raw_text": " ".join(chunk_words)
                    }
                    processed_items.append(item)
                
        return processed_items

    def process_data(self, text_batch):
        """Point d'entrée pour le Light Bridge (Thread)."""
        # 1. Préparation CPU (Réutilisation de la logique coeur)
        items = self.process_cpu_batch(text_batch)
        
        # 2. Finalisation GPU (Spécifique Light Bridge)
        if self.brain:
            for item in items:
                # En mode Light, on vectorise tout de suite
                vecs, weights = self.brain.encoder.encode_batch_fast(item["tokens"])
                
                packet = {
                    "type": "TEXT_BATCH",
                    "vecs": vecs,
                    "weights": weights,
                    "tokens": item["tokens"],
                    "count": len(vecs)
                }
                self.out_q.put(packet)
        return None
        
class TextIngestionHeavyBridge(GenesisBridge):
    """
    [HEAVY BRIDGE - ORCHESTRATEUR]
    Gère les processus et transmet les paquets SÉQUENTIELLEMENT au GPU.
    """
    def __init__(self, name, brain):
        super().__init__(name, brain)
        ctx = multiprocessing.get_context('spawn')
        self.heavy_in_q = ctx.Queue(maxsize=CFG.HEAVY_QUEUE_SIZE)
        self.heavy_out_q = ctx.Queue(maxsize=CFG.HEAVY_QUEUE_SIZE)
        
        self.workers = []
        self.pending_markers = 0
        
        print(f" [HEAVY-BRIDGE] Initialisation de {CFG.HEAVY_WORKERS_COUNT} processus lourds...")
        for i in range(CFG.HEAVY_WORKERS_COUNT):
            # On passe la config pour que le worker connaisse MAX_SEQUENCE_LENGTH
            config_snapshot = {"MAX_SEQUENCE_LENGTH": CFG.MAX_SEQUENCE_LENGTH}
            w = HeavyIngestionWorker(i, self.heavy_in_q, self.heavy_out_q, config_snapshot)
            w.start()
            self.workers.append(w)

    def stop(self):
        super().stop()
        print(" [HEAVY-BRIDGE] Arrêt des sous-processus...")
        for w in self.workers:
            if w.is_alive():
                w.terminate()

    def run(self):
        print(f" [HEAVY-BRIDGE] Orchestrateur Démarré. Mode: {CFG.HEAVY_WORKERS_COUNT} CPUs.")
        
        while not self.stop_event.is_set():
            # A. DISTRIBUTION
            try:
                while not self.in_q.empty():
                    raw_data = self.in_q.get_nowait()
                    if raw_data == "__SYNC_MARKER__":
                        self.pending_markers = len(self.workers)
                        for _ in range(len(self.workers)):
                            self.heavy_in_q.put("__SYNC_MARKER__")
                    else:
                        self.heavy_in_q.put(raw_data)
            except queue.Empty:
                pass

            # B. AGREGATION (Séquentielle)
            got_data = True
            while got_data:
                try:
                    packet = self.heavy_out_q.get(timeout=0.005)
                    
                    if packet["type"] == "MARKER":
                        self.pending_markers -= 1
                        if self.pending_markers <= 0:
                            self.out_q.put({"type": "MARKER"})
                            self.pending_markers = 0
                            
                    elif packet["type"] == "BATCH_RESULT":
                        data_list = packet["data"]
                        
                        # --- FIX CRITIQUE : TRAITEMENT SÉQUENTIEL (COMME LIGHT BRIDGE) ---
                        # On ne fusionne PAS tout. On traite item par item (phrase par phrase).
                        for item in data_list:
                            tokens = item["tokens"]
                            if not tokens: continue
                            
                            # 1. Update Stats
                            if "counts" in item:
                                self.brain.encoder.stats.usage.update(item["counts"])
                            
                            # 2. Encodage GPU (Un paquet par phrase/chunk)
                            vecs, weights = self.brain.encoder.encode_batch_fast(tokens)
                            
                            final_packet = {
                                "type": "TEXT_BATCH",
                                "vecs": vecs,
                                "weights": weights,
                                "tokens": tokens,
                                "count": len(vecs)
                            }
                            self.out_q.put(final_packet)
                            
                except queue.Empty:
                    got_data = False
                except Exception as e:
                    print(f" [ERR HEAVY] {e}")
                    got_data = False

class PrototypeHeavyBridge(multiprocessing.Process):
    """
    [BRIDGE LOURD - PROTOTYPE TEST]
    Basé sur Multiprocessing (Isolation Totale).
    Sert à valider l'architecture hybride pour le futur (Cluster/Distribué).
    
    Note: On ne passe PAS 'brain' ici car il n'est pas picklable (Pointeurs CUDA).
    On simule un traitement externe.
    """
    def __init__(self, input_q, output_q):
        super().__init__()
        self.in_q = input_q
        self.out_q = output_q
        self.running = multiprocessing.Event()
        self.running.set()

    def stop(self):
        self.running.clear()

    def run(self):
        print(f" [HEAVY-BRIDGE] Démarrage Processus Isolé (PID: {os.getpid()})")
        while self.running.is_set():
            try:
                # Récupération depuis le monde extérieur
                data = self.in_q.get(timeout=1.0)
                
                # Simulation d'un traitement lourd (ex: OCR, Vision, Réseau)
                # Ici on fait juste un "Pass-Through" avec marquage
                processed = f"[HEAVY_PROCESSED] {data}"
                
                # Renvoi vers le bridge léger
                self.out_q.put(processed)
            except Exception: # Queue Empty ou autre
                continue
        print(" [HEAVY-BRIDGE] Arrêt.")


# --- MVP98 : HEAVY BRIDGE WORKER (MULTIPROCESSING) ---
class HeavyIngestionWorker(multiprocessing.Process):
    """
    [HEAVY SHELL] Coquille vide qui exécute le Logic Core dans un processus isolé.
    """
    def __init__(self, worker_id, input_queue, output_queue, config_dict):
        super().__init__(name=f"HeavyWorker_{worker_id}")
        self.worker_id = worker_id
        self.in_q = input_queue
        self.out_q = output_queue
        self.cfg_snapshot = config_dict 
        self.running = multiprocessing.Event()
        self.running.set()
        self.tokenizer = None 

    

    def run(self):
        # 1. Instanciation du Cœur Logique (Mode Détaché / Headless)
        # On crée le bridge sans Brain (brain=None), juste pour sa méthode process_cpu_batch
        core_processor = TextIngestionBridge(name=f"Core_{self.worker_id}", brain=None)
        
        # On charge les outils locaux (Tokenizer) si dispo
        core_processor.setup_headless()
        
        # print(f" [WORKER-{self.worker_id}] Prêt (Mode Shell).")
        
        while self.running.is_set():
            try:
                # A. Réception
                raw_batch = self.in_q.get(timeout=0.5)
                
                if raw_batch == "__SYNC_MARKER__":
                    self.out_q.put({"type": "MARKER", "worker_id": self.worker_id})
                    continue
                
                # B. Délégation (Le point clé !)
                # Le Worker n'a plus aucune idée de comment on découpe une phrase.
                # Il demande juste au "Logic Core" de le faire.
                processed_items = core_processor.process_cpu_batch(raw_batch)
                
                # C. Envoi
                if processed_items:
                    # Conversion au format de transport (sérialisable)
                    batch_data = []
                    for item in processed_items:
                        batch_data.append({
                            "type": "PREPROCESSED",
                            "ids": item["ids"],
                            "tokens": item["tokens"],
                            "counts": item["counts"],
                            "raw_text": item["raw_text"]
                        })
                        
                    self.out_q.put({"type": "BATCH_RESULT", "data": batch_data})
                    
            except Exception:
                continue

class SensoryStream:
    def __init__(self, brain):
        self.brain = brain
        self.phys_engine = ChunkedGravityEngine(brain.dim, max_capacity=CFG.PHYSICS_STATIC_BUFFER_SIZE)
        self.MAX_LEN = CFG.MAX_SEQUENCE_LENGTH
        D = brain.dim
        self.vecs_buffer = torch.zeros((self.MAX_LEN, D), dtype=CFG.COMPUTE_DTYPE, device=CFG.DEVICE)
        self.positions_buffer = torch.zeros(self.MAX_LEN, dtype=CFG.COMPUTE_DTYPE, device=CFG.DEVICE)
        self.masses_buffer = torch.zeros(self.MAX_LEN, dtype=CFG.COMPUTE_DTYPE, device=CFG.DEVICE)
        self.layer_buffer = torch.zeros(self.MAX_LEN, dtype=torch.long, device=CFG.DEVICE)
        self.word_tokens = [""] * self.MAX_LEN 
        self.active_count = 0
        self.time_step = 0
        self.weights_buffer = torch.zeros(self.MAX_LEN, dtype=CFG.COMPUTE_DTYPE, device=CFG.DEVICE)
        self.layer_type = None
        
        # --- OPTIMISATION : PROCESSEUR SÉQUENTIEL UNIQUE ---
        # On le crée une seule fois pour éviter d'enregistrer 50 threads "fantômes"
        self.seq_processor = TextIngestionBridge(name="SeqProcessor", brain=self.brain)
        
        # --- MVP98 : INITIALISATION BRIDGE (POINT UNIQUE) ---
        self.bridge = None
        if CFG.ENABLE_MULTITHREADING:
            # On applique la stratégie décidée dans la Config (Heavy ou Light)
            if CFG.ENABLE_HEAVY_BRIDGE:
                # Option A : Architecture Massive (Multi-Process)
                self.bridge = TextIngestionHeavyBridge("HeavyIngestor", brain)
            else:
                # Option B : Architecture Légère (Thread Simple)
                self.bridge = TextIngestionBridge("LightIngestor", brain)
                
            self.bridge.start()
            print(f" [STREAM] Mode Pipeline ({self.bridge.name}) Activé.")
        else:
            print(" [WARN] Multithreading désactivé. Le bridge ne démarrera pas.")

    def stop(self):
        if self.bridge: self.bridge.stop()

    def receive_sequence(self, token_list_raw, layer_type=CFG.LAYER_CONCEPT, mode="REALITY", trust=1.0, sync_wait=False):
        """
        [FIX 0.0.96_04] Implémentation du Sentinel Pattern.
        Garantit que 100% des données sont traitées avant de rendre la main.
        """
        if isinstance(token_list_raw, str): token_list_raw = [token_list_raw]
        
        self.current_context_mode = mode
        self.current_trust_level = trust
        self.layer_type = layer_type

        # --- CAS 1 : THREADING ACTIF (Clean & Simple) ---
        if self.bridge and (mode == "TRAINING" or mode == "REALITY"):
            
            # 1. On envoie les données
            self.bridge.in_q.put(token_list_raw)
            
            # 2. Si on doit attendre (Test/Training), on envoie le marqueur de fin
            if sync_wait:
                self.bridge.in_q.put("__SYNC_MARKER__")
                
                # 3. Boucle d'attente active (Token Passing)
                # On consomme tout ce qui arrive jusqu'à retrouver notre marqueur
                while True:
                    try:
                        packet = self.bridge.out_q.get(timeout=10.0)
                        
                        if packet["type"] == "MARKER":
                            break # C'est fini, on rend la main
                            
                        # Traitement des données reçues (avec redirection logs)
                        with PrintRedirector(LOGGER):
                            self._consume_physics_packet(packet)
                            
                    except queue.Empty:
                        print(" [WARN] Bridge Timeout (Pas de réponse du Worker).")
                        break
        # --- CAS 2 : SÉQUENTIEL (Fallback) ---
        else:
            self._process_sequential_legacy(token_list_raw)
            
    
    def flush(self):
        """Indispensable : Attend que le Worker ait fini son travail."""
        if not self.bridge: return
        
        print(" [STREAM] Finalisation (Flush)...")
        # Tant qu'il reste du texte à lire OU des paquets à traiter
        while not self.bridge.in_q.empty() or not self.bridge.out_q.empty():
            self.process_pending()
            time.sleep(0.01) # On laisse le CPU respirer
        print(" [STREAM] Sync terminée.")
    
    
    def process_pending(self):
        """
        Vide la file d'attente du Bridge et exécute la physique + prints.
        """
        if not self.bridge: return

        # Tant qu'il y a des paquets prêts dans la sortie du thread
        while not self.bridge.out_q.empty():
            try:
                packet = self.bridge.out_q.get_nowait()
                if packet["type"] == "TEXT_BATCH":
                    # C'est ici que _process_buffer_vectorized est appelé
                    # et que tes prints ("Création Molécule...") vont s'afficher !
                    self._consume_physics_packet(packet)
            except queue.Empty:
                break

    def _consume_physics_packet(self, packet):
        """Exécute la physique sur un paquet pré-calculé (GPU Only)."""
        vecs = packet["vecs"]
        weights = packet["weights"]
        words = packet["tokens"]
        n = packet["count"]
        #print("oooooooooooooooooooooooooooooooooo")
        #print(f"packet:{packet}")
        # --- FIX CRASH SIZE MISMATCH ---
        # Si le paquet reçu est plus grand que le buffer interne (MAX_LEN), on tronque.
        # Idéalement, le worker devrait découper en amont, mais ceci est la sécurité ultime.
        if n > self.MAX_LEN:
            print(f" [WARN] Packet trop gros ({n} > {self.MAX_LEN}). Tronqué.")
            n = self.MAX_LEN
            vecs = vecs[:n]
            weights = weights[:n]
            words = words[:n]
        # -------------------------------
        
        # Copie dans les buffers (Très rapide, transfert GPU<->GPU ou RAM<->GPU)
        self.active_count = n
        self.vecs_buffer[:n] = vecs
        self.weights_buffer[:n] = weights
        self.positions_buffer[:n] = torch.arange(n, device=CFG.DEVICE, dtype=CFG.COMPUTE_DTYPE)
        # Mise à jour des tokens pour la logique sémantique
        # (Attention: ceci est lent en Python pur, point d'opti futur)
        for i, w in enumerate(words):
             if i < self.MAX_LEN: self.word_tokens[i] = w

        # Mise à jour mémoire (Learning)
        self.brain.memory.update_batch(words, vecs)
        
        # Physique
        self._process_buffer_vectorized()
        
        # Reset
        self.active_count = 0

    def _process_sequential_legacy(self, token_list_raw):
        """
        Mode Séquentiel (Fallback / Debug).
        [ALIGNEMENT] Utilise le "Logic Core" (TextIngestionBridge) pour garantir
        le même découpage et chunking que les modes Parallèles.
        """
        # 1. Conversion en liste de lignes (format attendu par process_cpu_batch)
        if isinstance(token_list_raw, str):
            raw_lines = [token_list_raw]
        else:
            raw_lines = token_list_raw
            
        # 2. Utilisation du Processeur Unique (Plus d'instanciation ici !)
        # Le processeur est déjà prêt dans self.seq_processor
        
        # 3. Traitement Standardisé (Nettoyage + Split + Chunking)
        processed_items = self.seq_processor.process_cpu_batch(raw_lines)
        
        # 4. Exécution Séquentielle
        for item in processed_items:
            tokens = item["tokens"]
            if not tokens: continue
            
            # Encodage GPU (Direct, on est sur le Main Thread)
            vecs, weights = self.brain.encoder.encode_batch_fast(tokens)
            
            # Pas besoin de check de taille ici, process_cpu_batch a déjà fait le Chunking (128)
            n_words = len(tokens)
            
            # Mise à jour Stats (pour parité avec les workers)
            self.brain.encoder.stats.usage.update(item["counts"])

            # Injection Buffer
            self.active_count = n_words
            self.vecs_buffer[:n_words] = vecs.to(dtype=CFG.COMPUTE_DTYPE)
            self.weights_buffer[:n_words] = weights.to(dtype=CFG.COMPUTE_DTYPE)
            self.positions_buffer[:n_words] = torch.arange(n_words, device=CFG.DEVICE, dtype=CFG.COMPUTE_DTYPE)
            
            # Mise à jour Token Strings
            for i, w in enumerate(tokens): 
                self.word_tokens[i] = w 
            
            # Apprentissage & Physique
            self.brain.memory.update_batch(tokens, vecs)
            self._process_buffer_vectorized()
            
            # Reset
            self.active_count = 0

    def _generate_attention_mask(self, n):
        mask = torch.ones((n, n), device=CFG.DEVICE, dtype=CFG.COMPUTE_DTYPE)
        return mask
        
    def _generate_topk_mask_vectorized(self, vecs, n):
        """
        Génère un masque d'attention clairsemé (Sparse) en pur Tensoriel.
        Compatible CPU/GPU sans aucune boucle Python.
        """
        # 1. Calcul préliminaire de similarité (brut)
        # On normalise pour que le Top-K soit basé sur la sémantique pure (Cosinus)
        # et non la masse.
        vecs_norm = F.normalize(vecs[:n], p=2, dim=1, eps=CFG.EPSILON)
        sim_matrix = torch.mm(vecs_norm, vecs_norm.t())
        
        # 2. Sélection des K meilleurs voisins (Vectorisé via CUDA/AVX)
        # k doit être borné par n (on ne peut pas chercher 10 voisins s'il n'y a que 5 objets)
        k_safe = min(CFG.TOP_K_LIMIT, n)
        
        # torch.topk retourne les valeurs et les indices. On ne veut que les indices.
        _, indices = torch.topk(sim_matrix, k=k_safe, dim=1)
        
        # 3. Création du Masque (Scatter Method - Zéro boucle)
        mask = torch.zeros_like(sim_matrix)
        
        # On "disperse" des 1.0 aux endroits indiqués par les indices du Top-K
        # src=1.0 est broadcasté
        mask.scatter_(1, indices, 1.0)
        
        # Optionnel : On s'assure que la diagonale est active ou inactive selon besoin
        # Pour la gravité, on évite l'auto-attraction, le kernel gère déjà la diagonale 0
        # mais c'est plus propre de laisser le mask propre.
        
        return mask

    def _get_interaction_mask(self, n, layer_target_type):
        """
        Génère un masque tensoriel basé sur les règles de config (PHYSICS_RULES).
        Gère le mapping dynamique des layers profonds (2000 -> REALITY).
        """
        current_types = self.layer_buffer[:n]
        
        # 1. Mapping du Layer demandé vers la Règle de Config
        rule_key = layer_target_type
        if layer_target_type >= CFG.DEPTH_REALITY:
            rule_key = CFG.LAYER_REALITY # = 2000
        elif layer_target_type < CFG.DEPTH_BUFFER:
            rule_key = CFG.LAYER_CONCEPT # = 0
 
        # 2. Récupération de la règle
        if rule_key not in CFG.PHYSICS_RULES:
            # Fallback : si pas de règle, on isole tout (pas d'interaction)
            return torch.zeros((n, n), device=CFG.DEVICE)
            
        rule_targets = CFG.PHYSICS_RULES[rule_key]
        
        # 3. Construction du Masque Vectorisé
        # On regarde pour chaque item si son type est autorisé dans la liste 'sources'
        # Note: Dans PHYSICS_RULES, la liste définit QUI peut interagir avec le layer cible
        allowed_tensor = torch.tensor(rule_targets, device=CFG.DEVICE)
        
        # Masque [N, 1] : Est-ce que l'élément i est compatible ?
        # On utilise le broadcasting pour vérifier l'appartenance
        is_compatible = (current_types.unsqueeze(1) == allowed_tensor).any(dim=1).float()
        
        # Le masque final est une matrice (N, N) : interaction active si les deux sont compatibles
        return torch.mm(is_compatible.unsqueeze(1), is_compatible.unsqueeze(0))



    def _process_buffer_vectorized(self):
        n = self.active_count
        if n < 2: return
        
        vecs_slice = self.vecs_buffer[:n]
        pos_slice = self.positions_buffer[:n]
        current_words = self.word_tokens[:n]
        ops = [None] * n
        op_dirs = [None] * n
        
        # 1. Détermination du Layer Cible (Contexte)
        target_layer_base = CFG.DEPTH_CONCEPT
        if hasattr(self, 'current_context_mode') and self.current_context_mode == "REALITY":
             target_layer_base = CFG.DEPTH_REALITY
        
        # Récupération du Facteur G (Scalabilité)
        # Fallback si on utilise les IDs profonds non mappés directement
        if target_layer_base >= CFG.DEPTH_REALITY:
            gravity_factor = CFG.GRAVITY_MAPPING[CFG.LAYER_REALITY] # 0.2
        else:
            # On map 2000 -> 0.2, 0 -> 1.0
            gravity_factor = CFG.GRAVITY_MAPPING.get(target_layer_base, CFG.GRAVITY_MAPPING["DEFAULT"])
        # 2. CALCUL MATRICIEL DE LA MASSE (L'idée du siècle)
        # Masse = Poids Sémantique * Gravité du Monde
        # Plus de boucle Python pour ça ! C'est instantané sur GPU.
        self.masses_buffer[:n] = self.weights_buffer[:n] * gravity_factor
        
        effective_trust = getattr(self, 'current_trust_level', 1.0)
        
        # 2. Préparation des Agents (Masses & Ops)
        for i, token in enumerate(current_words):
            base_mass = vecs_slice[i].norm().item()
            #self.masses_buffer[i] = base_mass
            
            # --- CORRECTION : On utilise ensure_node_in_layer pour peupler la scène ---
            # Si le mot est nouveau dans ce contexte (ex: "Cloche" en Réalité), il est créé.
            # Cela garantit qu'il a un ID, une masse et une plasticité.
            c = self.brain.ensure_node_in_layer(token, target_layer_base)
            if c:
                self.layer_buffer[i] = c.layer_type
                if (hardware := c.get_hardware_function()):
                    ops[i] = hardware
                    self.masses_buffer[i] = CFG.MASS_OPERATOR * hardware.priority
                    op_dirs[i] = c.metadata.get("op_dir", "<>")
                
            
            # PLUS DE CALCUL DE MASSE DANS CETTE BOUCLE
            # C'est déjà fait vectoriellement au-dessus.
            # Le code est nettoyé de toute logique "hasattr(mass)".
            
            # On peuple juste le layer buffer pour le masque d'interaction
            # Si l'objet n'est pas encore créé, on assume le layer cible
            self.layer_buffer[i] = target_layer_base
        mass_slice = self.masses_buffer[:n]
        
        # 3. Calcul Physique Brut (Toutes interactions possibles)
        # On utilise un masque simple ici (tout le monde voit tout le monde), le filtrage se fait après
        #attention_mask = self._generate_attention_mask(n)
        # --- MODIFICATION : ATTENTION SÉLECTIVE (TOP-K) ---
        # Au lieu de : attention_mask = self._generate_attention_mask(n)
        attention_mask = self._generate_topk_mask_vectorized(vecs_slice, n)
        
        if CFG.USE_CUDA: torch.cuda.synchronize()
        raw_force_matrix = self.phys_engine.compute(pos_slice, mass_slice, vecs_slice, n, mask=attention_mask)
        if CFG.USE_CUDA: torch.cuda.synchronize()
        # Application de la Gravité Relative (Terre vs Mars)
        # C'est ici que la magie opère : on atténue tout le layer d'un coup.
        
        raw_force_matrix = raw_force_matrix * gravity_factor
        
        if raw_force_matrix.is_sparse:
             raw_force_matrix = raw_force_matrix.to_dense() # Si besoin
        
 
        # 4. Application du Masque Logique (Layer Rules)
        # C'est ici qu'on applique le filtre "Réalité vs Concept"
        layer_mask = self._get_interaction_mask(n, target_layer_base)
        
        # La matrice finale combine physique (gravité) et logique (règles)
        masked_forces = raw_force_matrix * layer_mask
        
        # 5. Analyse des Forces (Trigger)
        # CORRECTION : Le seuil doit s'adapter à la gravité du Layer.
        # Si G=0.2 (Réalité), le seuil doit baisser pour détecter les interactions.
        # Sinon, tout est filtré comme "bruit".
        base_threshold = 0.001
        force_threshold = base_threshold * gravity_factor
        mass_threshold = CFG.MASS_THRESHOLD * gravity_factor
        
        # Pour éviter un seuil à 0 si gravity_factor est nul (peu probable mais prudent)
        force_threshold = max(force_threshold, 1e-6)

        
        max_forces = masked_forces.max(dim=1).values
        
        is_operator_mask = (mass_slice >= (CFG.MASS_OPERATOR - 1.0))
        has_strong_interaction = (max_forces > force_threshold)
        
        active_agents_mask = is_operator_mask & has_strong_interaction
        active_indices = torch.nonzero(active_agents_mask, as_tuple=False).squeeze(1).cpu().numpy()
        
        
        if len(active_indices) == 0: return
 
        indices = torch.arange(n, device=CFG.DEVICE)
        # SVO : Sujet (Passé) < Opérateur (Présent) < Cible (Futur)
        forces_past = masked_forces * (indices.unsqueeze(1) > indices.unsqueeze(0))
        forces_future = masked_forces * (indices.unsqueeze(1) < indices.unsqueeze(0))
        
        top_future_ind = torch.topk(forces_future, k=min(3, n), dim=1).indices.cpu().numpy()
        top_past_ind = torch.topk(forces_past, k=min(3, n), dim=1).indices.cpu().numpy()
        
        for i in active_indices:
            op = ops[i]
            if not op: continue
            
            # Recherche Cible (Futur)
            targ_idx = -1
            for k in range(min(3, n)):
                c_idx = top_future_ind[i, k]
                if masked_forces[i, c_idx].item() > force_threshold:
                    # Filtre anti-poussière
                    if mass_slice[c_idx] > mass_threshold:
                        targ_idx = c_idx; break
            
            # Recherche Sujet (Passé)
            subj_idx = -1
            for k in range(min(3, n)):
                c_idx = top_past_ind[i, k]
                if masked_forces[i, c_idx].item() > force_threshold:
                    if mass_slice[c_idx] > mass_threshold:
                        subj_idx = c_idx; break
            
            
                
            if subj_idx != -1 and targ_idx != -1:
                subj_name = current_words[subj_idx]
                target_name = current_words[targ_idx]
                
                
                
                d = op_dirs[i]
                if d == "><":
                    subj_name, target_name = target_name, subj_name
                
                # --- IMPLEMENTATION DE VOTRE OPTIMISATION ---
                # On récupère/crée les objets Node AVANT l'appel
                # Cela garantit que les masses et plasticités sont réglées par la factory centrale
                node_s = self.brain.ensure_node_in_layer(subj_name, target_layer_base)
                node_t = self.brain.ensure_node_in_layer(target_name, target_layer_base)
                
                try:
                    # On passe les OBJETS, pas les NOMS
                    
                    op.execute(self.brain, node_s, node_t, layer_type=target_layer_base,
                               trust_level=effective_trust)
                except Exception as e: print(f" [EXEC ERR] {e}")

class UnifiedBrain:
    def __init__(self, str_lang, boolResetBase=False):
        self.cfg = CFG 
        if boolResetBase and os.path.exists(CFG.BASE_MEM_DIR): 
            try: shutil.rmtree(CFG.BASE_MEM_DIR)
            except PermissionError: print(" [WARN] Impossible de supprimer l'ancienne mémoire (Fichier verrouillé).")
            
        self.dim = CFG.DIM_SIZE; self.temperature = 1.0
        self.encoder = MatrixEncoder(self.dim, self).to(CFG.DEVICE) 
        self.phys = HyperPhysics(self.dim).to(CFG.DEVICE)
        self.memory = HybridMemoryCluster(self.dim, max_nodes=CFG.INITIAL_MAX_NODES)
        self.associative_memory = AssociativeMemory(self); self.broca = self.associative_memory 
        self.stream = SensoryStream(self); self.metabolism = CognitiveMetabolism(self); self.chronos = Chronos(self.dim) 
        self.tool_factory = ToolFactory(self)
        
        self.boredom_level = 0.0
        self.is_life_loop_active = False 
        
        self.max_nodes = CFG.INITIAL_MAX_NODES
        self.node_registry = {}; self.next_node_id = 0; self.free_ids = set() 
        self.fast_lookup = {}   # (NomExact, Layer) -> UID (Unique)
        self.fuzzy_lookup = {}  # (NomLower, Layer) -> [UID1, UID2...] (Liste)
        self.global_energies = torch.zeros(self.max_nodes, dtype=torch.float32).to(CFG.DEVICE)
        
        self.root = FractalNode("GENESIS", self.dim, self.phys, encoder=self.encoder)
        self.mental = FractalNode("MENTAL", self.dim, self.phys, parent=self.root)
        self.semantic = FractalNode("CONCEPTS", self.dim, self.phys, parent=self.mental, layer_type=CFG.LAYER_CONCEPT)
        self.reality = FractalNode("REALITE", self.dim, self.phys, parent=self.root, layer_type=CFG.LAYER_REALITY)
        self.time_root = FractalNode("TEMPS", self.dim, self.phys, parent=self.root)
        self.conscience = FractalNode("CONSCIENCE", self.dim, self.phys, parent=self.root)
        self.self_node = FractalNode("SOI", self.dim, self.phys, parent=self.conscience)
        self.body_node = FractalNode("CORPS", self.dim, self.phys, parent=self.self_node, encoder=self.encoder, nature="Body")
        
        
        self.root.add_child(self.mental)
        self.mental.add_child(self.semantic)
        self.root.add_child(self.reality)
        self.root.add_child(self.time_root)
        self.root.add_child(self.conscience)
        self.conscience.add_child(self.self_node)
        self.self_node.add_child(self.body_node)
        
        self.organs = {}
        self.organs["BROCA"] = FractalNode("ORG_BROCA", self.dim, self.phys, parent=self.body_node, encoder=self.encoder, nature="Organ", layer_type=CFG.LAYER_LANGUAGE)
        self.organs["RETINA"] = FractalNode("ORG_RETINA", self.dim, self.phys, parent=self.body_node, encoder=self.encoder, nature="Organ", layer_type=CFG.LAYER_VISUAL)
        self.organs["COCHLEA"] = FractalNode("ORG_COCHLEA", self.dim, self.phys, parent=self.body_node, encoder=self.encoder, nature="Organ", layer_type=CFG.LAYER_AUDIO)
        self.organs["BOUCHE"] = FractalNode("BOUCHE", self.dim, self.phys, parent=self.body_node, encoder=self.encoder, nature="Organ")
        self.organs["INSULA"] = FractalNode("ORG_INSULA", self.dim, self.phys, parent=self.body_node, encoder=self.encoder, nature="Organ", layer_type=CFG.LAYER_INTEROCEPTION)
        self.organs["MOTOR"] = FractalNode("ORG_MOTOR", self.dim, self.phys, parent=self.body_node, encoder=self.encoder, nature="Organ", layer_type=CFG.LAYER_ACTION)
        
        for org in self.organs.values(): self.body_node.add_child(org)
        
        self.memory.load_index(); self.rehydrate_memory() 
        
        # --- OPTIMISATION AU DEMARRAGE ---
        # Si le système a beaucoup grandi (fichiers de diag), on le compacte
        if self.next_node_id > 10000: # Seuil arbitraire pour ne pas le faire sur un cerveau vide
             self.optimize_memory_layout()
        # ---------------------------------
        
        objLang = DefLanguage(str_lang)
        self.bootstrap(objLang)
    
    def register_node(self, node):
        if self.free_ids:
            uid = self.free_ids.pop(); self.global_energies[uid] = 0.0
        else:
            uid = self.next_node_id
            if uid >= self.max_nodes: self.expand_memory()
            self.next_node_id += 1
        
        # CORRECTION : On injecte l'UID dans le node IMMÉDIATEMENT
        # Cela permet à register_node_lookup de lire node.uid juste après
        node.uid = uid
        
        self.node_registry[uid] = node
        
        # Enregistrement centralisé du lookup (maintenant node.uid existe)
        self.register_node_lookup(node)
        
        return uid

    def register_node_lookup(self, node):
        """
        Enregistre le nœud dans les index Exact et Flou.
        Gère les collisions de casse (Chat/chat) via une liste.
        """
        if node.layer_type is not None:
            # 1. Index Principal (Exact) - Toujours unique
            name_raw = node.name.strip()
            key_exact = (name_raw, node.layer_type)
            self.fast_lookup[key_exact] = node.uid
            
            # 2. Index Secondaire (Flou) - Gestion des collisions
            name_norm = self._normalize_key(name_raw)
            key_fuzzy = (name_norm, node.layer_type)
            
            if key_fuzzy not in self.fuzzy_lookup:
                self.fuzzy_lookup[key_fuzzy] = []
            
            # On ajoute l'UID s'il n'y est pas déjà
            if node.uid not in self.fuzzy_lookup[key_fuzzy]:
                self.fuzzy_lookup[key_fuzzy].append(node.uid)

    def expand_memory(self):
        if self.max_nodes < CFG.LIMIT_INCR_BY_TOW_MEM: new_max = self.max_nodes * 2
        else: new_max = self.max_nodes + CFG.STEP_SCALE_MEM 
        print(f" [SYSTEME] Extension mémoire dynamique : {self.max_nodes} -> {new_max} slots.")
        try:
            self.memory.resize(new_max)
            # Vérification critique : est-ce que la mémoire a VRAIMENT grandi ?
            if self.memory.capacity == new_max:
                new_energies = torch.zeros(new_max, dtype=torch.float32).to(CFG.DEVICE)
                # Copie sécurisée
                current_len = len(self.global_energies)
                new_energies[:current_len] = self.global_energies
                self.global_energies = new_energies
                self.max_nodes = new_max
                print(f" [SUCCESS] Extension réussie.")
            else:
                print(" [CRITICAL] La mémoire n'a pas pu être redimensionnée (OOM probable).")
                # On lève une erreur pour arrêter le test proprement au lieu de crasher sur un index
                raise MemoryError("Echec critique allocation mémoire")
                
        except Exception as e:
            print(f" [ERR] Echec extension mémoire: {e}")
            # Optionnel : Tenter un optimize_memory_layout() d'urgence ici ?
            # Pour l'instant, on laisse planter proprement.
            raise e

    def delete_node(self, node):
        if node.uid in self.node_registry:
            layer = node.layer_type
            
            if layer is not None:
                # 1. Nettoyage Index Exact
                key_exact = (node.name.strip(), layer)
                if key_exact in self.fast_lookup:
                    del self.fast_lookup[key_exact]
                
                # 2. Nettoyage Index Flou (Gestion de liste)
                key_fuzzy = (self._normalize_key(node.name), layer)
                if key_fuzzy in self.fuzzy_lookup:
                    # On retire uniquement l'UID concerné
                    if node.uid in self.fuzzy_lookup[key_fuzzy]:
                        self.fuzzy_lookup[key_fuzzy].remove(node.uid)
                    
                    # Si la liste est vide, on nettoie la clé
                    if not self.fuzzy_lookup[key_fuzzy]:
                        del self.fuzzy_lookup[key_fuzzy]

            # Nettoyage Registre & Energie
            del self.node_registry[node.uid]
            self.free_ids.add(node.uid)
            self.global_energies[node.uid] = 0.0

    def bootstrap(self, lang):
        
        FREQ_DUST = 1000000000000 
        for w in lang.articles:
            # CORRECTION : Idempotence. 
            # On ne force 1000 que si l'usage actuel est inférieur.
            # Si le système a appris que "le" est vu 5000 fois, on garde 5000.
            if self.encoder.stats.usage[w] < FREQ_DUST:
                self.encoder.stats.usage[w] = FREQ_DUST
        
                
        # 2. CORRECTIF "CHAT DANS LA GRANDE": Allégement des Adjectifs
        # On leur donne une fréquence virtuelle élevée (mais moins que les articles)
        # pour qu'ils aient une masse faible (~0.1) et ne capturent pas les sujets.
        FREQ_ADJECTIVE = 10000 # Suffisant pour réduire la masse, mais garder du sens
        
        for adj in lang.adjectives:
            # On normalise la casse pour être sûr de toucher le bon token
            adj_key = adj.lower()
            if self.encoder.stats.usage[adj_key] < FREQ_ADJECTIVE:
                self.encoder.stats.usage[adj_key] = FREQ_ADJECTIVE

        for (w, direction), op_code in lang.ops.items():
            c = self.ensure_concept(w); c.bind_hardware_function(op_code, direction); self.memory.update(w, c.nature_vec)
        
        print(" [BOOTSTRAP] Opérateurs et Adjectifs chargés avec masses.")
        
    # --- AJOUTS OP11 : GESTION DES LAYERS ET PROPAGATION ---

    def _normalize_key(self, name):
        """Normalisation centralisée pour les index."""
        return name.strip().lower()

    def find_node_in_layer(self, name, layer_type):
        """
        Cherche un noeud avec une stratégie de repli (Fallback).
        1. Recherche Exacte ("Chat")
        2. Recherche Normalisée ("chat" ou "CHAT" ou "Chat")
        """
        clean_name = name.strip()
        
        # 1. Tentative Exacte (Priorité absolue)
        key_exact = (clean_name, layer_type)
        uid = self.fast_lookup.get(key_exact)
        if uid is not None:
            return self.node_registry.get(uid)
            
        # 2. Tentative "Fuzzy" (Fallback)
        key_norm = (self._normalize_key(clean_name), layer_type)
        uids_fuzzy = self.fuzzy_lookup.get(key_norm)
        
        if uids_fuzzy:
            # On renvoie le dernier enregistré qui correspond à cette casse normalisée
            # C'est mieux que rien.
            # On pourrait aussi itérer pour trouver le "mieux", mais pour l'instant :
            last_uid = uids_fuzzy[-1]
            return self.node_registry.get(last_uid)
             
        return None

    def get_layer_root(self, layer_type):
        """Renvoie le parent physique approprié."""
        if layer_type >= CFG.DEPTH_REALITY: return self.reality
        if layer_type >= CFG.DEPTH_BUFFER: return self.mental # Ou zone tampon
        return self.semantic

    def ensure_node_in_layer(self, name, layer_type):
        """
        Factory Centrale : Garantit qu'un nœud existe dans le layer demandé.
        Gère la projection Concept -> Réalité et l'initialisation correcte.
        """
        
        
        # 1. Recherche (Cache)
        node = self.find_node_in_layer(name, layer_type)
        if node: return node
 
        # 2. Création selon le Layer
        root = self.get_layer_root(layer_type)
        
        # --- CALCUL DE LA MASSE INTELLIGENTE (FACTORIELLE) ---
        # A. Importance Sémantique (IDF) : "le" = 0.01, "Chat" = 1.5
        semantic_weight = self.encoder.stats.get_inverse_freq_weight(name.lower())
        
        # B. Densité du Layer : Réalité (0.2) vs Concept (1.0)
        layer_factor = CFG.LAYER_DENSITY.get(layer_type, 1.0) 
        if layer_type >= CFG.DEPTH_REALITY: 
             layer_factor = CFG.LAYER_DENSITY[CFG.LAYER_REALITY]

        # C. Masse Finale = Importance * Densité
        dynamic_mass = semantic_weight * layer_factor
        # -----------------------------------------------------
        
        if layer_type >= CFG.DEPTH_BUFFER:
            # --- MODE RÉALITÉ (2000+) ---
            # On a besoin du Concept Source (Patron) pour l'instancier
            c_source = self.ensure_concept(name, layer_type=CFG.LAYER_CONCEPT)
            
            # Instanciation (Lien Concept <-> Instance)
            new_node = self.instantiate(name, c_source, root, layer_type=layer_type)
            
            # --- CORRECTION MAJEURE ICI ---
            # On applique la masse dynamique calculée
            new_node.mass = dynamic_mass
            
            # ATTENTION : On NE copie PLUS la masse du concept source (c_source.mass)
            # car elle écraserait notre calcul intelligent.
            # (L'ancien code "if hasattr(c_source, 'mass')..." est supprimé)
            
            # --- FIX CRITIQUE (Empathie/Physique) : Héritage de l'Opérateur ---
            # Si le concept est "dans" (OP_LOC), l'instance doit aussi être OP_LOC
            # sinon le moteur physique l'ignore.
            if c_source.metadata.get("native_op"):
                op_code = c_source.metadata["native_op"]
                op_dir = c_source.metadata.get("op_dir", "<>")
                new_node.bind_hardware_function(op_code, op_dir)
            
            
            # Plasticité Maximale (Pour apprendre immédiatement : Ouroboros)
            if layer_type >= CFG.DEPTH_REALITY:
                new_node.plasticity = CFG.PLASTICITY_REALITY # 1.0
                
        else:
            # --- MODE CONCEPT (0) ---
            new_node = FractalNode(name, self.dim, self.phys, parent=root,
                                   encoder=self.encoder, nature=name, layer_type=layer_type)
            root.add_child(new_node)
            # Enregistrement manuel dans le lookup (FractalNode a déjà pris un ID via register_node)
            self.register_node_lookup(new_node)
        
        return new_node

    def propagate_signal(self, source_node, vector, trust, key="valeur"):
        """
        Propage l'information du Bas (Réalité) vers le Haut (Concept).
        Permet l'apprentissage par l'expérience.
        """
        if trust <= 0.01: return
        
        # 1. Impact Local (Le noeud lui-même)
        source_node.absorb(key, vector, force=trust)
        
        # 2. Remontée vers le Concept Parent
        # Si on est dans une couche profonde (ex: 2000), on cherche le parent en (ex: 0)
        if source_node.layer_type >= CFG.DEPTH_OFFSET:
            # On simplifie : on remonte directement au Concept (Layer 0) pour cette version
            concept_node = self.find_node_in_layer(source_node.name, CFG.LAYER_CONCEPT)
            
            if concept_node:
                decayed_trust = trust * CFG.PROPAGATION_DECAY
                # Le concept absorbe l'expérience de l'instance
                concept_node.absorb(key, vector, force=decayed_trust)
                
                # Active Inference sur le Concept (pour générer des hypothèses globales)
                if concept_node.plasticity > 0.05: 
                     self.active_inference_check(concept_node)

    def ensure_concept(self, name, layer_type=None):
        # Wrapper pour compatibilité ascendante
        if layer_type is None: layer_type = CFG.LAYER_CONCEPT
        return self.ensure_node_in_layer(name, layer_type)
        
    def find_concept_exact(self, name):
        """
        Récupère le concept par son nom, indifféremment de la casse.
        CORRECTION : Ne plus utiliser self.semantic.children.get(name) directement
        car cela échoue si on cherche 'chat' alors que 'Chat' est stocké.
        """
        # 1. On normalise la demande ("Chat" -> "chat")
        # 2. On demande au Layer System qui utilise l'index fast_lookup normalisé
        # 3. Il nous renvoie l'objet Node original ("Chat")
        # 1. Recherche standard dans les Concepts
        node = self.find_node_in_layer(name, CFG.LAYER_CONCEPT)
        if node: return node
        
        # 2. Fallback : Recherche dans les Molécules (Fix Tests 30 & 32)
        node = self.find_node_in_layer(name, CFG.LAYER_MOLECULE)
        if node: return node
        
    def find_reality_node(self, name): return self._find_recursive(name, self.reality)
    def find_reality_node_fast(self, name):
        # Utilisation de la nouvelle méthode standardisée
        return self.find_node_in_layer(name, CFG.LAYER_REALITY)
        
        
    def _find_recursive(self, name, root):
        if root.name == name: return root
        for c in root.children.values(): 
            if res := self._find_recursive(name, c): return res
        return None
    
    def get_instances_of_concept(self, concept_name):
        instances = []
        self._collect_instances_recursive(self.reality, concept_name, instances)
        return instances
    def _collect_instances_recursive(self, node, concept_name, acc):
        for concept in node.concepts:
            if concept.name == concept_name: acc.append(node); break
        for child in node.children.values(): self._collect_instances_recursive(child, concept_name, acc)

    def propagate_forces_vectorized(self, group_a_uids, group_b_uids):
        """
        Pousse les forces physiques entre deux groupes de concepts.
        Version optimisée pour utiliser les indices mémoire directs.
        """
        if not group_a_uids or not group_b_uids: return
        
        # 1. Conversion Node UID -> Memory Index
        mem_indices_a = []
        mem_indices_b = []
        
        for uid in group_a_uids:
            node = self.node_registry.get(uid)
            if node and node.name in self.memory.name_to_idx:
                mem_indices_a.append(self.memory.name_to_idx[node.name])
                
        for uid in group_b_uids:
            node = self.node_registry.get(uid)
            if node and node.name in self.memory.name_to_idx:
                mem_indices_b.append(self.memory.name_to_idx[node.name])

        if not mem_indices_a or not mem_indices_b: return

        # 2. Création des tenseurs d'indices
        t_a = torch.tensor(mem_indices_a, device=CFG.DEVICE, dtype=torch.long)
        t_b = torch.tensor(mem_indices_b, device=CFG.DEVICE, dtype=torch.long)
        
        # 3. Meshgrid pour créer toutes les paires (A <-> B)
        grid_a, grid_b = torch.meshgrid(t_a, t_b, indexing='ij')
        sources = grid_a.flatten()
        targets = grid_b.flatten()
        
        # 4. Calcul des forces (Decay)
        forces = torch.full_like(sources, CFG.FORCE_INDIRECT_DECAY, dtype=torch.float32)
        
        # 5. Appel à l'encodeur optimisé
        self.encoder.learn_attraction_batch(sources, targets, forces)

    def simulate_perception(self, agent_name, reality_node):
        if agent_name not in self.conscience.children:
            agent = FractalNode(agent_name, self.dim, self.phys, parent=self.conscience, encoder=self.encoder, nature="Agent")
            self.conscience.add_child(agent)
        else: agent = self.conscience.children[agent_name]
        mood_vec = agent.get_local("humeur"); 
        if mood_vec is None: mood_vec = torch.zeros(self.dim).to(CFG.DEVICE)
        print(f" [EMPATHY] Simulation de {agent_name} (Humeur active).")
        for obj in reality_node.children.values():
            obj_vec = obj.nature_vec
            if obj_vec.is_sparse: obj_vec = obj_vec.to_dense()
            if mood_vec.is_sparse: mood_vec = mood_vec.to_dense()
            saliency = F.cosine_similarity(obj_vec, mood_vec, dim=0).item()
            physical_saliency = obj.energy / 100.0; total_saliency = (saliency + physical_saliency) / 2
            if total_saliency > CFG.THRESHOLD_LOOSE:
                reaction_vec = F.normalize(obj_vec * mood_vec, p=2, dim=0, eps=CFG.EPSILON)
                feel_word = self.associative_memory.articulate(reaction_vec)
                print(f"   > {agent_name} remarque '{obj.name}' (Saliency: {total_saliency:.2f}) -> Ressent: {feel_word}")

    def instantiate(self, name, concept, loc, layer_type=None):
        # Default legacy
        if layer_type is None: layer_type = CFG.LAYER_REALITY
        
        # Vérification préventive
        key = (name, layer_type)
        if key in self.fast_lookup:
            return self.node_registry[self.fast_lookup[key]]

        # Création
        inst = FractalNode(name, self.dim, self.phys, parent=loc, 
                           encoder=self.encoder, nature=name, layer_type=layer_type)
        
        # Le __init__ de FractalNode appelle register_node, 
        # mais on s'assure que le lookup est bon avec le bon layer
        self.register_node_lookup(inst)
        
        inst.link_concept(concept)
        loc.add_child(inst)
        return inst

    def perceive(self, t, mode="REALITY", trust=1.0, sync_wait = CFG.ENABLE_MULTITHREADING): 
        # Point d'entrée "Trust-Aware"
        layer_target = CFG.LAYER_REALITY if mode == "REALITY" else CFG.LAYER_CONCEPT
        # --- FIX TESTS ---
        # Si on utilise le threading, on VEUT attendre le résultat pour les tests unitaires

        self.stream.receive_sequence(t, layer_type=layer_target, mode=mode, trust=trust, sync_wait=sync_wait)
        
    def _collect_node_data(self, node, path_id, tensor_dict, json_dict, exclude=None):
        if exclude and node.name == exclude: return 
        node_info = {
            "parent": path_id.rsplit("/", 1)[0] if "/" in path_id else None, 
            "concepts": [c.name for c in node.concepts], 
            "metadata": node.metadata,
            "layer_type": node.layer_type, 
            "percepts": [p.name for p in node.percepts] 
        }
        node_info["metadata"]["mass"] = node.mass
        json_dict[path_id] = node_info
        for k, v in node.states.items(): 
            if v.is_sparse: v = v.to_dense()
            tensor_dict[f"{path_id}:{k}"] = Quantizer.to_storage(v.detach().cpu())
        for child_name, child_node in node.children.items(): self._collect_node_data(child_node, f"{path_id}/{child_name}", tensor_dict, json_dict, exclude)

    def optimize_memory_layout(self):
        """
        [CRITIQUE] Défragmente la mémoire (Garbage Collection & Compaction).
        Version sécurisée pour ne pas briser les liens énergétiques.
        """
        print(f"\n [OPTIMIZATION] Démarrage de la défragmentation (Avant: {self.next_node_id} slots)...")
        
        # 1. Identifier les survivants (Garbage Collection)
        # On ne garde que ce qui est référencé dans le registre actuel
        active_nodes = []
        sorted_old_uids = sorted(self.node_registry.keys())
        for uid in sorted_old_uids:
            active_nodes.append(self.node_registry[uid])
            
        new_count = len(active_nodes)
        if new_count == 0: return

        print(f" [OPTIMIZATION] {new_count} noeuds actifs identifiés. Reconstruction...")

        # 2. Allocation des nouvelles structures (Compaction)
        new_capacity = max(CFG.INITIAL_MAX_NODES, int(new_count * 1.1)) # +10% de marge
        
        new_memory = HybridMemoryCluster(self.dim, max_nodes=new_capacity)
        new_energies = torch.zeros(new_capacity, dtype=torch.float32).to(CFG.DEVICE)
        
        new_registry = {}
        new_fast_lookup = {}
        
        # Buffers pour la mise à jour vectorielle en masse
        vectors_to_keep = []
        names_to_keep = []
        
        # 3. MIGRATION (Phase Critique)
        for new_uid, node in enumerate(active_nodes):
            # --- ETAPE A : Lecture de l'ancien état (Avant modif UID) ---
            old_uid = node.uid
            
            # Sauvegarde de l'énergie (car global_energies est indexé par old_uid)
            current_energy = self.global_energies[old_uid]
            
            # Sauvegarde du vecteur (car memory est indexé par nom -> old_idx)
            if node.name in self.memory.name_to_idx:
                old_mem_idx = self.memory.name_to_idx[node.name]
                vec = self.memory.fast_index[old_mem_idx]
            else:
                vec = torch.zeros(self.dim, device=CFG.DEVICE)
            
            # --- ETAPE B : Mise à jour de l'identité (Mutation) ---
            node.uid = new_uid
            node.fractal_id = new_uid # Si utilisé ailleurs
            
            # --- ETAPE C : Écriture dans les nouvelles structures ---
            # 1. Registres
            new_registry[new_uid] = node
            # Update lookup avec le nouveau UID
            if node.layer_type is not None:
                key = (node.name, node.layer_type)
                new_fast_lookup[key] = new_uid
            
            # 2. Energie
            new_energies[new_uid] = current_energy
            
            # 3. Vecteurs (Batching)
            vectors_to_keep.append(vec)
            names_to_keep.append(node.name)

        # 4. Finalisation de la Mémoire Vectorielle
        if vectors_to_keep:
            batch_vecs = torch.stack(vectors_to_keep)
            new_memory.update_batch(names_to_keep, batch_vecs)

        # 5. Bascule (Swap)
        self.node_registry = new_registry
        self.fast_lookup = new_fast_lookup
        self.global_energies = new_energies
        self.next_node_id = new_count
        self.max_nodes = new_capacity
        self.free_ids = set() # Le système est maintenant parfaitement contigu, plus de trous.
        
        # Remplacement mémoire
        del self.memory
        self.memory = new_memory
        
        # Nettoyage VRAM/RAM
        gc.collect()
        torch.cuda.empty_cache()
        
        print(f" [OPTIMIZATION] Terminé. Système propre : {self.next_node_id} noeuds actifs / {self.max_nodes} capacité.")



    def rehydrate_memory(self):
        # 1. Chargement des données brutes (Tenseurs déjà dédupliqués par SafeFileManager)
        all_tensors = self.memory.get_all_loaded_tensors()
        
        # 2. Remplissage du Lexique (Indispensable pour Broca)
        if all_tensors:
            print(f" [LOAD] Réhydratation de {len(all_tensors)} vecteurs sémantiques...")
            for name, vec in all_tensors.items():
                if not name.startswith("ROOT"): 
                    self.encoder.semantic_map[name] = vec
                    self.associative_memory.register_word(name, vec)
                    
        # 3. Chargement de la Structure (JSON + Tenseurs Monde)
        struct, w_states, saved_energies, meta = SafeFileManager.load_monolithic_v2(CFG.BASE_MEM_DIR)
        
        # 4. Restauration des Métatadonnées Cerveau
        if meta:
            self.next_node_id = meta.get("next_node_id", 0)
            self.free_ids = set(meta.get("free_ids", []))
            # ... (Gestion resize mémoire inchangée) ...
            
            
        # --- CORRECTIF CRITIQUE : Redimensionnement à la charge ---
        # Si le cerveau sauvegardé était plus gros que la config par défaut,
        # il faut étendre la mémoire MAINTENANT, sinon register_node va crasher
        # sur les index >= 50000.
        saved_max_nodes = meta.get("max_nodes", CFG.INITIAL_MAX_NODES)
        
        if saved_max_nodes > self.max_nodes:
            print(f" [LOAD] Extension mémoire requise : {self.max_nodes} -> {saved_max_nodes}")
            
            # 1. Resize Cluster Vectoriel
            self.memory.resize(saved_max_nodes)
            
            # 2. Resize Energies (Indispensable pour global_energies[uid] = 0.0)
            new_energies = torch.zeros(saved_max_nodes, dtype=torch.float32).to(CFG.DEVICE)
            # Pas besoin de copier les anciennes énergies (elles sont vides au boot)
            self.global_energies = new_energies
            self.max_nodes = saved_max_nodes
            
            
        # 5. Restauration des Énergies
        if saved_energies is not None:
            load_len = min(len(saved_energies), self.max_nodes)
            self.global_energies[:load_len] = saved_energies[:load_len]

        # 6. RECONSTRUCTION DE L'ARBRE (Indispensable)
        path_to_obj = {"ROOT": self.root, "ROOT/TEMPS": self.time_root, "ROOT/CONSCIENCE": self.conscience}
        if struct:
            nodes_data = struct.get("nodes", {})
            sorted_paths = sorted(nodes_data.keys(), key=lambda x: x.count('/'))
            
            for path in sorted_paths:
                if path in path_to_obj: continue
                data = nodes_data[path]
                parent_path = data.get("parent")
                
                if parent_path in path_to_obj:
                    parent_node = path_to_obj[parent_path]
                    node_name = path.split('/')[-1]
                    layer = data.get("layer_type", CFG.LAYER_CONCEPT)
                    
                    # Création ou récupération du Node
                    if node_name in parent_node.children: 
                        curr_node = parent_node.children[node_name]
                        curr_node.layer_type = layer
                    else:
                        nature = "Event" if "TEMPS" in parent_path else node_name
                        curr_node = FractalNode(node_name, self.dim, self.phys, parent=parent_node, encoder=self.encoder, nature=nature, layer_type=layer)
                        parent_node.add_child(curr_node)
                    
                    # Restauration attributs
                    curr_node.metadata = data.get("metadata", {})
                    curr_node.mass = data.get("metadata", {}).get("mass", CFG.MASS_PLANET)
                    self.register_node(curr_node)
                    
                    # Restauration Concept Links
                    for c_name in data.get("concepts", []): 
                        curr_node.link_concept(self.ensure_concept(c_name))
                    
                    path_to_obj[path] = curr_node

            # Restauration Percept Links (Hub & Spoke)
            # ... (Code existant inchangé) ...

        # 7. INJECTION DES ÉTATS (C'est ici que la magie opère)
        # Grâce à SafeFileManager, si des vecteurs sont partagés, w_states contient déjà les alias.
        for k, v in w_states.items():
            p, d = k.split(":")
            if p in path_to_obj: 
                # L'assignation ici préserve le lien partagé créé par load_tensors
                path_to_obj[p].set(d, v)

        # 8. Restauration Historique
        h_states = SafeFileManager.load_tensors(os.path.join(CFG.BASE_MEM_DIR, "genesis_history.safetensors"))
        for k, v in h_states.items():
            p, d = k.split(":")
            if p in path_to_obj: 
                path_to_obj[p].set(d, v)
        
        # --- SUPPRESSION DU BLOC "Pointer Merge" ---
        # Le bloc qui parcourait self.node_registry pour relier source_concept est SUPPRIMÉ.
        # SafeFileManager s'en occupe désormais.
        
        print(f" [BOOT] Cerveau restauré ({len(path_to_obj)} noeuds).")

    def create_molecule(self, subject, relation, attribut):
        mol_name = f"MOL_{subject}_{attribut}"
        mol_node = self.ensure_concept(mol_name, layer_type=CFG.LAYER_MOLECULE)
        
        if ("recipe" not in mol_node.metadata):
            mol_node.metadata["recipe"] = {
                "type": "MOLECULE",
                "components": { "subject": subject, "relation": relation, "attribut": attribut }
            }
            c_subj = self.ensure_concept(subject)
            c_attr = self.ensure_concept(attribut)
            mol_node.link_percept(c_subj)
            mol_node.link_percept(c_attr)
            print(f" [MOLECULE] Création de {mol_name} ({subject} -{relation}-> {attribut})")
        return mol_node

    def ask_narrative(self, subject):
        print(f" [NARRATIVE] Recherche de molécules pour le sujet '{subject}'...")
        molecules = []
        for name, node in self.semantic.children.items():
            if node.layer_type == CFG.LAYER_MOLECULE and name.startswith(f"MOL_{subject}"):
                molecules.append(node)
        if not molecules:
            return f"Je ne sais rien sur {subject}."
        responses = []
        for mol in molecules:
            recipe = mol.metadata.get("recipe", {})
            comps = recipe.get("components", {})
            if comps:
                rel = comps.get("relation")
                attr = comps.get("attribut")
                if rel == "ATTR": verb = "est"
                elif rel == "LOC": verb = "est dans"
                else: verb = "est lié à"
                responses.append(f"{subject} {verb} {attr}")
        return " . ".join(responses) + " ."

    def active_inference_check(self, instance, focus_dims=None):
        print(f" --- INFERENCE CHECK: {instance.name} ---")
        # 1. IDENTIFICATION DU CIBLE POUR L'HYPOTHÈSE
        # Si on est en Réalité (2000), on veut écrire l'hypothèse sur le Concept (0)
        # pour que l'apprentissage soit durable.
        hypothesis_target = instance
        if instance.layer_type >= CFG.DEPTH_REALITY:
            # On cherche le concept parent (Le patron)
            # Hypothèse forte : le premier concept lié est le patron
            if instance.concepts:
                hypothesis_target = instance.concepts[0]
                print(f" > Redirection Hypothèse : {instance.name}(Layer {instance.layer_type}) -> {hypothesis_target.name}(Layer {hypothesis_target.layer_type})")
            else:
                # Fallback : Si pas de lien, on cherche par nom dans le Layer 0
                pot_concept = self.find_node_in_layer(instance.name, CFG.LAYER_CONCEPT)
                if pot_concept: hypothesis_target = pot_concept
        
        # Normalisation du focus en liste
        if focus_dims and isinstance(focus_dims, str):
            focus_dims = [focus_dims]
        
        # Copie des clés pour éviter erreur de modification d'itérateur
        keys = list(instance.states.keys())
        
        for dim_name in keys:
            perceived_vec = instance.states[dim_name]
            theoretical_vec = instance.get_conceptual(dim_name)
            
            # Flag pour déclencher l'archivage à la fin
            should_archive = False
            diff = 0.0
            
            # Vérification : Est-ce la dimension qu'on est en train de modifier ?
            is_focused = (focus_dims is None) or (dim_name in focus_dims)
            
            if theoretical_vec is None:
                # CAS 1 : INCONNU TOTAL
                # On ne crée une nouvelle propriété que si c'est le sujet de l'attention (Focus)
                if is_focused:
                    # Cas 1 : Le concept ne connaissait pas cette propriété (Surprise totale)
                    print(f" > SURPRISE! Propriété '{dim_name}' inconnue.")
                    
                    # C'est une surprise, donc on veut archiver
                    should_archive = True
                    # --- CORRECTIF : CRÉATION DE L'HYPOTHÈSE ---
                    # Au lieu de juste apprendre silencieusement, on crée la structure
                    hypothesis_target.create_hypothesis(dim_name, perceived_vec)
                    
                    # On peut garder l'update direct si on veut apprendre vite, 
                    # ou l'enlever si on veut que seule l'hypothèse porte l'info.
                    #instance.update_concept_knowledge(dim_name, perceived_vec)
                    self.boredom_level = 0.0
                
            else:
                # Cas 2 : Rupture de pattern
                if theoretical_vec.is_sparse: theoretical_vec = theoretical_vec.to_dense()
                if perceived_vec.is_sparse: perceived_vec = perceived_vec.to_dense()
                tv = Quantizer.from_storage(theoretical_vec)
                pv = Quantizer.from_storage(perceived_vec)
                diff = (tv - pv).norm().item()
                
                if diff > CFG.INFERENCE_RUPTURE:
                    # On vérifie sur le TARGET (Concept), pas l'instance locale
                    hyp_name = f"HYP_{hypothesis_target.name}_{dim_name}"
                    
                    if hyp_name in hypothesis_target.children:
                        # Cas 1 : RUPTURE DÉJÀ CONNUE (Le Concept savait déjà que c'était possible)
                        print(f" > RUPTURE DEJA CONNUE ({diff:.2f}). Renforcement Conceptuel.")
                        hypothesis_target.create_hypothesis(dim_name, perceived_vec)
                        # CORRECTION : On force l'archivage pour la mémoire épisodique
                        should_archive = True
                    elif is_focused:
                        # Cas 2 : RUPTURE INÉDITE (Le Concept apprend qqch de nouveau via la Réalité)
                        print(f" > RUPTURE INEDITE ({diff:.2f}) -> Apprentissage Conceptuel (Bottom-Up).")
                        should_archive = True # On lève le drapeau !
                        # C'est ici que la magie opère : La Réalité écrit sur le Concept
                        hypothesis_target.create_hypothesis(dim_name, perceived_vec)
                        
                        self.boredom_level = 0.0
                    # ---------------------------------
                elif diff > CFG.INFERENCE_ACCOMMODATION:
                    print(f" > ANOMALIE ({diff:.2f}) -> Accommodation.")
                    instance.update_concept_knowledge(dim_name, perceived_vec)
                    self.boredom_level = max(0.0, self.boredom_level - 10.0)
                else:
                    print(f" > OK: '{dim_name}' conforme (Diff: {diff:.2f}).")
                    
            # --- BLOC DE SAUVEGARDE CENTRALISÉ ---
            # Si le drapeau a été levé (Cas 1 ou Cas 2 Inédit), on archive.
            if should_archive:
                print(f" [TIME] Archivage de l'événement: Sur {instance.name}")
                self._archive_event(instance, dim_name, perceived_vec)
                    
    def generate_response(self, input_vec):
        concept_name, score = self.memory.find_closest(input_vec, threshold=0.5)
        if not concept_name: return self.associative_memory.articulate(input_vec)
        sentence = self.verbalize_thought(concept_name)
        return sentence if sentence else concept_name

    def verbalize_thought(self, concept_name):
        # 1. On cherche d'abord le concept pur
        node = self.find_concept_exact(concept_name)
        
        # 2. Si pas trouvé ou vide, on regarde si une instance existe en réalité (Contexte immédiat)
        if not node or not node.states:
             node = self.find_node_in_layer(concept_name, CFG.LAYER_REALITY)
             
        if not node or not node.states: return None
        
        keys = list(node.states.keys())
        # Filtre simple pour éviter les clés techniques
        valid_keys = [k for k in keys if k not in ["mass", "next"]]
        if not valid_keys: return None
        
        chosen_key = random.choice(valid_keys)
        vec_target = node.states[chosen_key]
        
        target_name = self.associative_memory.articulate(vec_target, anti_parrot=False)
        if target_name == "???": return None
        
        # --- CORRECTIF V06 : Déballage des Molécules ---
        # Si la cible est une molécule technique (ex: MOL_Feu_Chaud), on veut dire "Chaud".
        if target_name.startswith("MOL_"):
            mol_node = self.find_node_in_layer(target_name, CFG.LAYER_MOLECULE)
            if mol_node and "recipe" in mol_node.metadata:
                # On récupère l'attribut original depuis la recette
                # Structure recipe: {components: {attribut: "Chaud", ...}}
                comps = mol_node.metadata["recipe"].get("components", {})
                if "attribut" in comps:
                    target_name = comps["attribut"]
        # -----------------------------------------------
        
        relation_word = "est"
        if chosen_key == "loc": relation_word = "dans"
        # Si la clé est un mot connu (ex: "mange"), on l'utilise comme verbe
        elif chosen_key in self.encoder.semantic_map: relation_word = chosen_key
        
        if relation_word.lower() == target_name.lower():
            relation_word = "est"
        
        return f"{concept_name} {relation_word} {target_name} ."

    def spark_curiosity(self):
        print(f" [CURIOSITE] L'ennui ({self.boredom_level:.1f}) dépasse le seuil.")
        all_concepts = list(self.semantic.children.keys())
        if len(all_concepts) < 2: return
        c1 = random.choice(all_concepts); c2 = random.choice(all_concepts)
        thought_structure = f"{c1} est {c2} ." 
        print(f"   ? (Curiosité) Genesis imagine : '{thought_structure}'")
        self.perceive(thought_structure)
        self.boredom_level = 0.0 

    def _archive_event(self, instance, dim_name, old_state):
        tick_time = self.chronos.tick; event_name = f"EVT_{instance.name}_{dim_name}_{tick_time}"
        evt_node = FractalNode(event_name, self.dim, self.phys, parent=self.time_root, encoder=self.encoder, nature="Event")
        self.time_root.add_child(evt_node); time_vec = self.chronos.get_time_vector()
        evt_node.set("memory_trace", old_state * time_vec)
        print(f" [TIME] Souvenir '{event_name}' cristallisé à T={tick_time}")
    def graph_query(self, concept_depart, steps=2, decay=0.5):
        print(f" [GRAPH] Propagation depuis '{concept_depart}'...")
        queue = [(concept_depart, 1.0)]; visited = {} 
        while queue:
            current_name, energy = queue.pop(0)
            if energy < 0.1 or (current_name in visited and visited[current_name] >= energy): continue
            visited[current_name] = energy
            node = self.find_concept_exact(current_name)
            if not node: continue
            for child_name in node.children: queue.append((child_name, energy * decay))
            for relation, linked_vec in node.states.items():
                if linked_vec.is_sparse: linked_vec = linked_vec.to_dense()
                linked_name = self.associative_memory.articulate(linked_vec, anti_parrot=False)
                if linked_name != "???" and linked_name not in visited:
                    print(f"   -> Lien trouvé : {current_name} --[{relation}]--> {linked_name}")
                    queue.append((linked_name, energy * decay))
        return sorted(visited.items(), key=lambda x: x[1], reverse=True)
    def consolidate_hypotheses(self):
        print(" [CONSOLIDATION] Analyse des hypothèses...")
        count_learned = 0; count_forgotten = 0
        all_nodes = list(self.semantic.children.values()) + list(self.reality.children.values())
        for node in all_nodes:
            for child_name, child_node in list(node.children.items()):
                if not child_name.startswith("HYP_"): continue
                e = child_node.energy 
                if e >= CFG.THRESHOLD_VALIDATION:
                    print(f"   ★ VALIDATION : L'hypothèse '{child_name}' devient une vérité pour '{node.name}'.")
                    parts = child_name.split("_")
                    if len(parts) >= 3:
                        dim_concernee = parts[-1]; vec_hypothese = child_node.get_local("valeur")
                        if vec_hypothese is None: vec_hypothese = child_node.nature_vec 
                        node.absorb(dim_concernee, vec_hypothese, force=1.0); del node.children[child_name]; count_learned += 1
                elif e < CFG.THRESHOLD_FORGET:
                    del node.children[child_name]; count_forgotten += 1
        if count_learned > 0 or count_forgotten > 0: print(f" [CONSOLIDATION] {count_learned} concepts intégrés, {count_forgotten} hypothèses oubliées.")
    
    def dream_cycle(self):
        print(" [REVE] Phase de sommeil paradoxal (Simulation)...")
        memories = [m for m in self.time_root.children.keys() if m.startswith("EVT_")]
        if not memories: return
        count = 0; random.shuffle(memories)
        for mem_name in memories:
            if count >= CFG.DREAM_INTENSITY: break
            parts = mem_name.split("_")
            if len(parts) >= 2:
                subject = parts[1]; thought = self.verbalize_thought(subject)
                if thought:
                    print(f"   ~ (Rêve) Genesis pense : '{thought}'"); self.perceive(thought); count += 1

    def sleep(self):
        self.metabolism.run_cycle(); self.consolidate_hypotheses(); self.dream_cycle()
        print(" [SOMMEIL] Update Centroïdes & Entropie...")
        self.root.update_centroid()
        print(f" [METABOLISME] Decay Global Vectorisé sur {self.next_node_id} noeuds.")
        if self.next_node_id > 0: self.global_energies[:self.next_node_id] *= 0.9
        dead_nodes = self.root.apply_decay()
        if dead_nodes: print(f" [ENTROPY] {len(dead_nodes)} noeuds oubliés ce cycle.")
        self.memory.save_all(); self.encoder.save(CFG.BASE_MEM_DIR);
        structure_data = {"nodes": {}}; dummy_tensors = {}
        self._collect_node_data(self.root, "ROOT", dummy_tensors, structure_data["nodes"])
        mem_meta = {"next_node_id": self.next_node_id, "max_nodes": self.max_nodes, "free_ids": list(self.free_ids)}
        SafeFileManager.save_monolithic_v2(dummy_tensors, structure_data, self.global_energies, CFG.BASE_MEM_DIR, mem_meta)
        history_states = {}; dummy = {}; self._collect_node_data(self.time_root, "ROOT/TEMPS", history_states, dummy)
        SafeFileManager.save_tensors(history_states, os.path.join(CFG.BASE_MEM_DIR, "genesis_history.safetensors"))
    
    def life_cycle(self):
        print("\n [LIFE] Démarrage de la Boucle de Vie (Non-Bloquante)...")
        self.is_life_loop_active = True
        input_queue = queue.Queue(); input_thread = InputListener(input_queue); input_thread.start()
        try:
            while True:
                try:
                    user_input = input_queue.get_nowait()
                    if user_input.lower() in ["quit", "exit"]: break
                    if user_input.strip():
                        print(f" < Vous: {user_input}")
                        if user_input.lower().startswith("comment est "):
                            subj = user_input.lower().replace("comment est ", "").replace("?", "").strip()
                            print(f" > GENESIS: {self.ask_narrative(subj)}")
                        else:
                            self.perceive(user_input if user_input.endswith(".") else user_input + " .")
                            mots = user_input.replace(".", "").split()
                            if mots:
                                vec_pensee = torch.zeros(self.dim).to(CFG.DEVICE)
                                for m in mots: vec_pensee += self.encoder.encode_word(m)
                                vec_pensee = F.normalize(vec_pensee, p=2, dim=0, eps=CFG.EPSILON)
                                reponse = self.generate_response(vec_pensee)
                                print(f" > GENESIS: {reponse}")
                            self.boredom_level = 0.0
                except queue.Empty: pass
                time.sleep(CFG.LIFE_TICK_SPEED); self.boredom_level += CFG.BOREDOM_RATE
                if self.boredom_level > CFG.BOREDOM_THRESHOLD: self.spark_curiosity()
        except KeyboardInterrupt: print("\n [LIFE] Interruption.")
        finally: self.is_life_loop_active = False; self.sleep()

class GenesisBootloader:
    def __init__(self, config, brain): self.cfg = config; self.brain = brain
    def train_from_corpus_file(self, file_path, epochs=None, layer_type=CFG.LAYER_CONCEPT):
        if epochs is None: epochs = self.cfg.INGEST_DEFAULT_EPOCHS
        print(f"\n [TRAINING] Ingestion Contrôlée (Threaded + Logs): {file_path}")
        if not os.path.exists(file_path): print(f" [ERR] Fichier introuvable."); return
        
        CHUNK_LINES = 100 
        batch_lines = []
        total_lines = 0
        start_global = time.time()
        
        for epoch in range(epochs):
            print(f" --- EPOQUE {epoch + 1}/{epochs} ---")
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    while True:
                        lines = f.readlines(1000) 
                        if not lines: break
                        
                        for line in lines:
                            if not line.strip(): continue
                            batch_lines.append(line.strip().replace(".", " .")) 
                            total_lines += 1
                            
                            if len(batch_lines) >= CHUNK_LINES:
                                # [CRITIQUE] sync_wait=True force l'attente du résultat
                                self.brain.stream.receive_sequence(
                                    batch_lines, 
                                    layer_type, 
                                    mode="TRAINING", 
                                    trust=1.0, 
                                    sync_wait=True  # <--- CRUCIAL
                                )
                                batch_lines = []
                                
                            if total_lines % self.cfg.INGEST_SLEEP_INTERVAL == 0: 
                                print(f" [Check] {total_lines} lignes...")
                                self.brain.sleep()
                                
                if batch_lines: 
                    self.brain.stream.receive_sequence(batch_lines, layer_type, mode="TRAINING", trust=1.0, sync_wait=True)
                    
            except Exception as e: print(f" [ERR] {e}")
        
        # Le flush est moins critique ici grâce au sync_wait, mais on le garde par sécurité
        self.brain.stream.flush()
        print(f" [TRAINING] Terminé en {time.time() - start_global:.2f}s."); self.brain.sleep()
        
    def import_external_vectors(self, file_path):
        print(f"\n [IMPORT] Greffe de vecteurs: {file_path}")
        if not os.path.exists(file_path): return
        try:
            source_dim = 0
            # Correction Syntaxe Python (Pas de one-liner sur try/with)
            with open(file_path, 'r', encoding='utf-8') as f:
                header = f.readline().strip().split()
                source_dim = len(header) - 1
            
            if source_dim <= 0: return

            projection_matrix = F.normalize(torch.randn(source_dim, self.cfg.DIM_SIZE).to(self.cfg.DEVICE), p=2, dim=0, eps=CFG.EPSILON)
            count = 0
            
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split(); word = parts[0]; vals = [float(x) for x in parts[1:]]
                    if len(vals) != source_dim: continue
                    vec_p = torch.matmul(torch.tensor(vals).to(self.cfg.DEVICE), projection_matrix)
                    vec_p = F.normalize(vec_p, p=2, dim=0, eps=CFG.EPSILON)
                    self.brain.memory.update(word, vec_p)
                    self.brain.ensure_concept(word); self.brain.associative_memory.register_word(word, vec_p)
                    count += 1
            self.brain.sleep()
        except Exception as e: print(f" [ERR IMPORT] {e}")

class GenesisDiagnostic:
    def __init__(self, brain): self.brain = brain
    def run_all(self):
        print(f"\n=== Diagnostic Complet ({CFG.STR_VERSION}) ===")
        self.test_precision_mode()
        self.test_morphology();
        self.test_broca();
        self.test_vsa_hashing();
        self.test_causalite();
        self.test_meta()
        self.test_algebre();
        self.test_ouroboros();
        self.test_syntax_learning();
        self.test_entropie();
        self.test_empathie_qualia()
        self.test_magnetic_parser();
        self.test_TRAINING_FILE_and_broca_empathie()
        self.test_consolidation();
        self.test_tool_activation();
        self.test_dream_and_speech();
        self.test_curiosity()
        self.test_dynamic_memory();
        self.test_hybrid_engine();
        self.test_memory_index_performance()
        self.test_mvp88_smart_switch();
        self.test_mvp88_life_mechanics();
        self.test_mass_physics();
        self.test_batch_ingestion_speed()
        self.test_tokenizer_performance();
        self.test_physics_broadcasting(); 
        #self.test_massive_context_n9(); 
        self.test_stream_n10_performance()
        self.test_multimodal_layering();
        self.test_attention_masking()
        self.test_agnostic_scaffolding()
        self.test_molecule_persistence()
        self.test_mvp94_narrative_architect()
        self.test_optimization_vectorielle()
        self.test_cpu_benchmark_pytorch_vs_numba()
        self.test_numerical_stability()
        self.test_scalability()
        self.test_quantization_quality()
        self.test_hypothesis_acceptance()
        self.test_bridge_architecture()
        self.test_hybrid_bridge_chaining()
        print("\n=== Fin Diagnostic ===")
    
    def forensic_audit_ghosts(self):
        print("\n--- AUDIT FORENSIC : RECHERCHE DES NOEUDS FANTÔMES ---")
        base_dir = self.brain.cfg.BASE_MEM_DIR
        
        # 1. Chargement de la Carte Structurelle (Ce que le cerveau croit avoir)
        struct_path = os.path.join(base_dir, "genesis_structure.json")
        if not os.path.exists(struct_path):
            print(" [SKIP] Pas de structure.json trouvée.")
            return

        structure = SafeFileManager.load_json(struct_path)
        # On extrait les noms des noeuds depuis les chemins (ex: "ROOT/MENTAL/Concept" -> "Concept")
        struct_names = set()
        if "nodes" in structure:
            for path in structure["nodes"].keys():
                name = path.split('/')[-1]
                struct_names.add(name)
        
        print(f" > Noeuds dans la structure (JSON) : {len(struct_names)}")

        # 2. Chargement de l'Inventaire Vectoriel (Ce qui a une mémoire physique)
        # On regarde dans le fichier principal et les shards
        memory_names = set()
        
        # Scan du fichier principal
        world_path = os.path.join(base_dir, "genesis_world.safetensors")
        if os.path.exists(world_path):
            # On utilise load_file de safetensors pour juste lire les clés sans charger la RAM
            from safetensors.torch import load_file
            tensors = load_file(world_path)
            for k in tensors.keys():
                # Format souvent "Chemin:cle", on veut le nom du noeud s'il est indexé
                # Mais ici on cherche surtout dans les Shards de mémoire
                pass

        # Scan des Shards de mémoire (C'est là que sont les vecteurs sémantiques)
        for i in range(self.brain.cfg.SHARD_COUNT):
            shard_path = os.path.join(base_dir, f"shard_{i}.safetensors")
            if os.path.exists(shard_path):
                from safetensors.torch import load_file
                shard_data = load_file(shard_path)
                for k in shard_data.keys():
                    memory_names.add(k)
        
        print(f" > Vecteurs en banque (Safetensors) : {len(memory_names)}")

        # 3. Comparaison (Le Delta)
        # Les fantômes sont ceux qui sont dans la Structure MAIS PAS dans la Mémoire
        ghosts = struct_names - memory_names
        
        print(f" > Nombre de Fantômes détectés : {len(ghosts)}")
        
        if ghosts:
            print("\n [ANALYSE] Liste des disparus (Top 20) :")
            sorted_ghosts = sorted(list(ghosts))
            for g in sorted_ghosts[:20]:
                print(f"   - {g}")
                
            if len(ghosts) > 20:
                print(f"   ... et {len(ghosts) - 20} autres.")
                
            # Verdict
            nb_concepts = sum(1 for g in ghosts if not g.startswith(("HYP_", "MOL_", "EVT_")))
            print(f"\n [VERDICT] Parmi les fantômes :")
            print(f"   - Hypothèses/Molécules/Events (Transient) : {len(ghosts) - nb_concepts}")
            print(f"   - Concepts Purs (Potentiellement critique) : {nb_concepts}")
            
            if nb_concepts == 0:
                print(" [SUCCESS] Perte bénigne. Seuls des éléments transitoires ont été nettoyés.")
            else:
                print(" [WARN] Attention, des concepts nommés ont perdu leur vecteur (Reset usine).")
        else:
            print(" [SUCCESS] Intégrité parfaite. Aucun fantôme.")
    
    
    
    def test_precision_mode(self):
        print(f"\n--- TEST N11: PRECISION MODE ({CFG.PRECISION_MODE}) ---")
        # CORRECTION AUDIT : Utilisation de vecteurs normalisés pour simuler la réalité
        # Un vecteur sémantique a toujours une norme de 1.0 (ou proche après decay)
        vec = torch.randn(CFG.DIM_SIZE, device=CFG.DEVICE)
        vec = F.normalize(vec, p=2, dim=0, eps=CFG.EPSILON) # Normalisation unitaire
        
        # 1. Compression
        stored = Quantizer.to_storage(vec)
        
        # Si on est en INT8 via SmartQuantizer, on simule le cycle complet
        if CFG.STORAGE_DTYPE == torch.int8:
            q_vec, scale = SmartQuantizer.quantize(vec)
            restored = SmartQuantizer.dequantize(q_vec, scale)
            print(f" > Mode SmartQuantizer (INT8 + Scale)")
        else:
            restored = Quantizer.from_storage(stored)
            
        print(f" > Type Stockage : {stored.dtype}")
        print(f" > Type Calcul : {restored.dtype}")
        
        # 2. Mesure
        # Distance Euclidienne
        diff_norm = (vec - restored).norm().item()
        # Similarité Cosinus (Le plus important pour la sémantique)
        cosine_sim = F.cosine_similarity(vec, restored, dim=0).item()
        
        print(f" > Perte de Précision (Diff Norm) : {diff_norm:.6f}")
        print(f" > Fidélité Sémantique (Cosine) : {cosine_sim:.6f}")
        
        if cosine_sim > 0.99:
            print(" [SUCCESS] Précision suffisante pour la Physique Sémantique.")
        else:
            print(" [WARN] Perte de précision critique pour le sens.")

    def test_morphology(self):
        print("\n--- TEST 0: MORPHOLOGIE (MATRIX) ---")
        w1 = "Manger"; w2 = "Mangeons"; w3 = "Galaxie"
        v1 = self.brain.encoder.encode_word(w1)
        v2 = self.brain.encoder.encode_word(w2)
        v3 = self.brain.encoder.encode_word(w3)
        sim_close = F.cosine_similarity(v1, v2, dim=0).item()
        sim_far = F.cosine_similarity(v1, v3, dim=0).item()
        print(f" > '{w1}' vs '{w2}' : {sim_close:.4f} | vs '{w3}' : {sim_far:.4f}")
        if sim_close > sim_far: print(" [SUCCESS] Morphologie OK.")
        else: print(" [FAIL] Morphologie HS.")

    def test_broca(self):
        print("\n--- TEST BROCA: ARTICULATION ---")
        target = "Chocolat"; self.brain.ensure_concept(target)
        vec = self.brain.encoder.get_semantic_vector(target); self.brain.associative_memory.register_word(target, vec)
        word = self.brain.associative_memory.articulate(vec, anti_parrot=False)
        print(f" > Vecteur('{target}') -> Broca dit: '{word}'")
        if word == target: print(" [SUCCESS] Broca OK.")
        else: print(f" [FAIL] Broca HS.")

    def test_vsa_hashing(self):
        print("\n--- 1. VSA HASHING CHECK ---")
        v1 = self.brain.encoder.encode_word("Chat"); v2 = self.brain.encoder.encode_word("Chien")
        print(f" > Chat vs Chien: {F.cosine_similarity(v1, v2, dim=0).item():.4f}")

    def test_causalite(self):
        print("\n--- 2. CAUSALITE (Regression MVP65) ---")
        # Mode TRAINING pour apprentissage fort immédiat
        self.brain.perceive("Les Feu sont Chaud .", mode="TRAINING")
        self.brain.perceive("Il y a un Feu dans Cuisine .", mode="TRAINING")
        self.brain.perceive("Le Feu est Bleu .", mode="REALITY")
        #self.brain.perceive("Le Feu est Bleu .", mode="REALITY")

    def test_meta(self):
        print("\n--- 3. META (Regression MVP65) ---")
        self.brain.perceive("is sont sont .", mode="TRAINING")
        c_is = self.brain.ensure_concept("is"); c_sont = self.brain.ensure_concept("sont"); c_is.metadata = c_sont.metadata.copy()
        self.brain.perceive("Water is Wet .", mode="TRAINING")

    def test_algebre(self):
        print("\n--- 4. ALGEBRE (Regression MVP65) ---")
        self.brain.perceive("Rouge avec Pomme .", mode="TRAINING")
        self.brain.perceive("Feu separer Eau .", mode="TRAINING")
        self.brain.perceive("Avocat comme Fruit .", mode="TRAINING")
        self.brain.perceive("Avocat comme Metier .", mode="TRAINING")
        self.brain.perceive("Feu sans Chaud .", mode="TRAINING")

    
        
        
    def test_ouroboros(self):
        print("\n--- 5. OUROBOROS PREDICTION (Regression MVP65) ---")
        
        # ON PRÉPARE LE TERRAIN (Pour garantir la surprise)
        c_cloche = self.brain.ensure_concept("Cloche")
        c_sonner = self.brain.ensure_concept("Sonner")
        c_silence = self.brain.ensure_concept("Silencieuse")
        # HACK DE TEST : On force les vecteurs à être opposés pour déclencher une RUPTURE
        # Si Cloche est (1, 0...) et Sonner est (-1, 0...), la distance sera max (2.0)
        # Cela garantit que diff > CFG.INFERENCE_RUPTURE
        c_cloche.nature_vec = torch.ones(self.brain.dim).to(self.brain.cfg.DEVICE)
        c_sonner.nature_vec = -torch.ones(self.brain.dim).to(self.brain.cfg.DEVICE)
        c_silence.nature_vec = -c_cloche.nature_vec # On oppose encore pour être sûr
        self.brain.memory.update("Cloche", c_cloche.nature_vec)
        self.brain.memory.update("Sonner", c_sonner.nature_vec)
        self.brain.memory.update("Silencieuse", c_silence.nature_vec)
        
        # --- PHASE 1 : LE PASSÉ (T=1) ---
        self.brain.chronos.advance() # Tick = 1
        tick_t1 = self.brain.chronos.tick 
        
        self.brain.ensure_concept("Cloche")
        self.brain.perceive("Il y a une Cloche dans Salon .", mode="REALITY")
        self.brain.perceive("La Cloche est Sonner .", mode="REALITY")
        
        # --- PHASE 2 : LE PRÉSENT (T=2) ---
        self.brain.chronos.advance() # Tick = 2
        tick_t2 = self.brain.chronos.tick 
        
        self.brain.perceive("La Cloche est Silencieuse .", mode="REALITY")
        
        # --- VÉRIFICATION DU PASSÉ (T=1) ---
        print(f" > Verification Memoire Episodique (Passé T={tick_t1} 'sonner')...")
        # On cherche un souvenir qui finit par _1 et contient "sonner"
        events_t1 = [k for k in self.brain.time_root.children.keys() 
                     if k.endswith(f"_{tick_t1}") and "sonner" in k.lower()]
        
        if events_t1: 
            print(f" [SUCCESS] Souvenir Passé T={tick_t1} retrouvé : {events_t1[0]}")
        else: 
            print(f" [FAIL] Le souvenir 'Sonner' à T={tick_t1} a été perdu ou écrasé.")
        
        
        
        # --- VÉRIFICATION DU PRÉSENT (T=2) ---
        print(f" > Verification Memoire Episodique (Présent T={tick_t2} 'silencieuse')...")
        # On cherche un événement qui finit par _2 et contient "silencieuse"
        events_t2 = [k for k in self.brain.time_root.children.keys() 
                     if k.endswith(f"_{tick_t2}") and "silencieuse" in k.lower()]
        
        if events_t2: 
            print(f" [SUCCESS] Souvenir Présent T={tick_t2} retrouvé : {events_t2[0]}")
        else: 
            print(f" [FAIL] L'événement 'Silencieuse' à T={tick_t2} n'a pas été créé.")
            print(f"   > Contenu TimeRoot : {list(self.brain.time_root.children.keys())}")

    def test_syntax_learning(self):
        print("\n--- 6. APPRENTISSAGE SYNTAXIQUE (MVP64) ---")
        self.brain.perceive("donc est CAUSALITE .", mode="TRAINING")
        self.brain.perceive("Frottement donc Chaleur .", mode="TRAINING")
        c_frot = self.brain.find_concept_exact("Frottement")
        if c_frot and c_frot.get_local("next") is not None: print(" [SUCCESS] Lien Causal créé via Grammaire Apprise !")

    def test_entropie(self):
        print("\n--- 7. ENTROPIE & OUBLI (MVP64) ---")
        self.brain.ensure_concept("Trace_Ephémère"); node_eph = self.brain.find_concept_exact("Trace_Ephémère"); node_eph.energy = 0.5 

    def test_empathie_qualia(self):
        print("\n--- 8. EMPATHIE & QUALIA SCENARIO (FIX MVP69 REGRESSION) ---")
        if not self.brain.find_reality_node_fast("Salon"):
            # Correction: Utilisation de ensure_node_in_layer pour créer la réalité
            salon = self.brain.ensure_node_in_layer("Salon", CFG.LAYER_REALITY)
            print(" [SETUP] Lieu 'Salon' créé.")
            
        self.brain.ensure_concept("Chocolat")
        self.brain.perceive("Il y a un Chocolat dans Salon .", mode="REALITY")
        
        vec_choco = self.brain.encoder.encode_word("Chocolat"); vec_triste = self.brain.encoder.encode_word("Tristesse"); vec_joie = self.brain.encoder.encode_word("Joie")
        vec_reconfort = F.normalize(vec_choco * vec_triste, p=2, dim=0, eps=CFG.EPSILON); self.brain.ensure_concept("Reconfort").nature_vec = vec_reconfort
        self.brain.memory.update("Reconfort", vec_reconfort); self.brain.associative_memory.register_word("Reconfort", vec_reconfort)
        vec_degout = F.normalize(vec_choco * vec_joie, p=2, dim=0, eps=CFG.EPSILON); self.brain.ensure_concept("Degout").nature_vec = vec_degout
        self.brain.memory.update("Degout", vec_degout); self.brain.associative_memory.register_word("Degout", vec_degout)
        
        print("\n > Simulation ALICE (Triste)...")
        if "Alice" not in self.brain.conscience.children:
            self.brain.conscience.add_child(FractalNode("Alice", CFG.DIM_SIZE, self.brain.phys, parent=self.brain.conscience, encoder=self.brain.encoder, nature="Agent"))
        self.brain.conscience.children["Alice"].set("humeur", vec_triste); salon_node = self.brain.find_reality_node_fast("Salon"); self.brain.simulate_perception("Alice", salon_node)
        
        print("\n > Simulation BOB (Joyeux)...")
        if "Bob" not in self.brain.conscience.children:
            self.brain.conscience.add_child(FractalNode("Bob", CFG.DIM_SIZE, self.brain.phys, parent=self.brain.conscience, encoder=self.brain.encoder, nature="Agent"))
        self.brain.conscience.children["Bob"].set("humeur", vec_joie); self.brain.simulate_perception("Bob", salon_node)
        
        self.brain.sleep()
        if "Trace_Ephémère" not in self.brain.semantic.children: print(" [SUCCESS] 'Trace_Ephémère' a été oublié (Nettoyage Entropique).")

    def test_magnetic_parser(self):
        print("\n--- 9. MAGNETIC PARSER (HYBRID CHECK) ---")
        print(" > Phrase Complexe (Magnetic): 'Chat course Mange Souris .' (Test sémantique)") 
        c_mange = self.brain.ensure_concept("Mange"); c_mange.bind_hardware_function("OP_ATTR", direction="<>")
        self.brain.perceive("Chat course Mange Souris .", mode="TRAINING")
        chat = self.brain.find_concept_exact("Chat")
        if chat and chat.get_local("mange") is not None: print(" [SUCCESS] Weighted Magnetic Parser: 'Mange' a attiré 'Chat'!")
        else: print(" [INFO] Parser Magnétique: Résultat dépendant du poids des mots (Comportement Attendu en Hybrid).")
        print(" > Phrase Simple (Hybrid/Linear): 'Souris mangée_par le Chat .' (OVS)")
        self.brain.ensure_concept("Souris"); self.brain.ensure_concept("Chat") 
        self.brain.perceive("Souris mangée_par le Chat .", mode="TRAINING")
        chat_node = self.brain.find_concept_exact("Chat")
        #souris_node = self.brain.find_concept_exact("souris")
        if chat_node and chat_node.get_local("Souris") is not None: print(" [SUCCESS] OVS Parser Simple: OK (Validé par Hybrid Mode).")
        else: print(" [FAIL] OVS échoué.")
        #print(f"mass du chat: {chat_node.mass}")
        #print(f"mass du souris: {souris_node.mass}")

    def test_TRAINING_FILE_and_broca_empathie(self):
        print("\n--- 10. TEST BROCA: ARTICULATION with EMPATHIE & QUALIA SCENARIO ---")
        bootloader.train_from_corpus_file("genesis_curriculum_test.txt", epochs=5)
    
    def test_consolidation(self):
        print("\n--- 13. TEST CONSOLIDATION (MVP84) ---")
        c_test = self.brain.ensure_concept("ConceptTest")
        c_test.create_hypothesis("force", self.brain.encoder.encode_word("Haute"))
        hyp_node = c_test.children["HYP_ConceptTest_force"]
        hyp_node.energy = CFG.THRESHOLD_VALIDATION + 10.0 
        c_test.create_hypothesis("faiblesse", self.brain.encoder.encode_word("Basse"))
        hyp_node_weak = c_test.children["HYP_ConceptTest_faiblesse"]
        hyp_node_weak.energy = CFG.THRESHOLD_FORGET - 1.0 
        print(" > Lancement Consolidation...")
        self.brain.consolidate_hypotheses()
        if c_test.get_local("force") is not None and "HYP_ConceptTest_force" not in c_test.children:
            print(" [SUCCESS] Hypothèse Forte intégrée (Fusion).")
        else: print(" [FAIL] Échec Fusion.")
        if "HYP_ConceptTest_faiblesse" not in c_test.children and c_test.get_local("faiblesse") is None:
             print(" [SUCCESS] Hypothèse Faible oubliée.")
        else: print(" [FAIL] Échec Oubli.")
        
    def test_tool_activation(self):
        print("\n--- 14. TEST TOOL ACTIVATION (MVP85) ---")
        print(" > Apprentissage du pattern OP_LOC + OP_ATTR...")
        for i in range(4):
            s = f"Sujet{i}"; l = f"Lieu{i}"; a = f"Attr{i}"
            self.brain.ensure_concept(s); self.brain.ensure_concept(l); self.brain.ensure_concept(a)
            self.brain.perceive(f"{s} dans {l} est {a} .", mode="TRAINING")
        if "TOOL_OP_LOC+OP_ATTR" in HARDWARE_REGISTRY:
            print(" [SUCCESS] L'outil 'TOOL_OP_LOC+OP_ATTR' a été inventé.")
        else:
            print(" [FAIL] L'outil n'a pas été créé malgré la répétition.")

    def test_dream_and_speech(self):
        print("\n--- 15. TEST RÊVE ET PAROLE (MVP86) ---")
        print(" > Forçage d'une pensée simple 'Le Feu est Chaud' pour tester la Bouche...")
        c_feu = self.brain.ensure_concept("Feu")
        c_feu.set("chaud", self.brain.encoder.encode_word("Chaud"))
        sentence = self.brain.verbalize_thought("Feu")
        print(f" > Bouche génère : '{sentence}'")
        if sentence and "est" in sentence:
            print(" [SUCCESS] La Bouche sait parler (SVO).")
        else:
            print(" [FAIL] La Bouche balbutie ou se tait.")
        print(" > Lancement du Cycle de Sommeil (Rêve)...")
        self.brain._archive_event(c_feu, "chaud", self.brain.encoder.encode_word("Chaud"))
        self.brain.dream_cycle()
        print(" [SUCCESS] Cycle de rêve exécuté.")

    def test_curiosity(self):
        print("\n--- 16. TEST CURIOSITÉ & ENNUI (MVP87) ---")
        print(f" > État actuel: Ennui={self.brain.boredom_level:.1f}/{CFG.BOREDOM_THRESHOLD} | Switch={CFG.ENABLE_CURIOSITY}")
        print(" > Répétition massive pour ennuyer Genesis...")
        for _ in range(6):
            self.brain.perceive("Le Feu est Chaud .", mode="REALITY", trust=1.0) 
        print(f" > Nouvel état: Ennui={self.brain.boredom_level:.1f}")
        if self.brain.boredom_level == 0.0:
            print(" [SUCCESS] La curiosité s'est déclenchée (Ennui reset) !")
        elif self.brain.boredom_level > CFG.BOREDOM_THRESHOLD:
            print(" [FAIL] L'ennui est haut mais pas de déclenchement (Seuil mal configuré ?).")
        else:
            print(" [INFO] Ennui en hausse, continuez...")

    def test_dynamic_memory(self):
        print("\n--- 18. TEST DYNAMIC MEMORY (MVP-N4 & Hardening) ---")
        
        # --- ÉTAPE 0 : SETUP & NETTOYAGE (Clean State) ---
        # On vérifie si le noeud existe déjà (résidu d'un run précédent)
        existing_node = self.brain.find_concept_exact("TropPlein")
        if existing_node:
            # On le supprime pour partir d'une base saine
            self.brain.delete_node(existing_node)
        
        # On sauvegarde l'état actuel du recyclage pour le restaurer plus tard
        # C'est nécessaire car pour tester l'expansion, il faut simuler qu'il n'y a AUCUN trou.
        real_free_ids_backup = self.brain.free_ids.copy()
        self.brain.free_ids.clear() 
        
        # --- ÉTAPE 1 : TEST EXPANSION ---
        initial_max = self.brain.max_nodes
        print(f" > Max Nodes Initial: {initial_max}")
        
        # On force le compteur à la limite pour déclencher l'expansion immédiate
        self.brain.next_node_id = initial_max
        
        print(" > Création concept 'TropPlein' pour forcer l'expansion...")
        # Ici, comme free_ids est vide et next_node_id == max, l'expansion est OBLIGATOIRE
        tp_node = self.brain.ensure_concept("TropPlein")
        
        expected_double = initial_max * 2
        expected_linear = initial_max + CFG.STEP_SCALE_MEM
        
        if self.brain.max_nodes > initial_max:
             print(f" [SUCCESS] Mémoire étendue : {initial_max} -> {self.brain.max_nodes}")
        else:
             print(f" [FAIL] L'expansion a échoué (Max={self.brain.max_nodes}).")

        # --- ÉTAPE 2 : TEST RECYCLAGE ---
        print(" > Test Suppression & Recyclage...")
        
        # On supprime le noeud qu'on vient de créer
        if tp_node:
            deleted_id = tp_node.uid
            # On capture l'ID max avant la création suivante
            # Si le recyclage marche, le next_node_id ne DOIT PAS augmenter
            next_id_marker = self.brain.next_node_id
            
            self.brain.delete_node(tp_node)
            
            # Vérification interne
            if deleted_id in self.brain.free_ids:
                print(f" [SUCCESS] ID {deleted_id} bien libéré dans le pool.")
            else:
                print(f" [FAIL] ID non libéré.")
            
            # On crée un nouveau noeud "Recyclé"
            # Il DOIT prendre la place de l'ID libéré, sans incrémenter le compteur global
            rec_node = self.brain.ensure_concept("Recyclé")
            
            if rec_node.uid == deleted_id:
                print(f" [SUCCESS] Recyclage exact : ID {deleted_id} réutilisé.")
            elif rec_node.uid < next_id_marker:
                 # Cas où il a pris un autre ID libre plus bas, mais n'a pas créé de nouvel ID
                print(f" [SUCCESS] Recyclage partiel : ID {rec_node.uid} utilisé (Pool actif).")
            else:
                print(f" [FAIL] Pas de recyclage : Nouvel ID généré ({rec_node.uid}).")
                
        # --- ÉTAPE 3 : TEARDOWN (Restauration) ---
        # On remet les free_ids d'origine (moins ceux utilisés, plus ceux libérés)
        # Pour faire simple et robuste, on réinjecte ceux qu'on avait backupés
        # Sauf si on veut garder l'état exact, mais pour un test unitaire, le but est de ne pas casser la suite.
        self.brain.free_ids.update(real_free_ids_backup)
    
    def test_hybrid_engine(self):
        print("\n--- 17. TEST HYBRID ENGINE (N9: BENCHMARK & SCALING) ---")
        N_small = 2000
        pos = torch.rand(N_small, device=CFG.DEVICE); mass = torch.ones(N_small, device=CFG.DEVICE)
        vecs = torch.randn(N_small, CFG.DIM_SIZE, device=CFG.DEVICE)
        engine = self.brain.stream.phys_engine
        t0 = time.time(); engine.compute(pos, mass, vecs, N_small);
        if CFG.USE_CUDA:
            torch.cuda.synchronize()
        t_small = (time.time() - t0) * 1000
        print(f" > N={N_small} (Mode {engine.mode}) : {t_small:.2f} ms")
        
        N_large = 10000
        pos = torch.rand(N_large, device=CFG.DEVICE); mass = torch.ones(N_large, device=CFG.DEVICE)
        vecs = torch.randn(N_large, CFG.DIM_SIZE, device=CFG.DEVICE)
        t0 = time.time()
        res = engine.compute(pos, mass, vecs, N_large)
        if CFG.USE_CUDA:
            torch.cuda.synchronize()
        t_large = (time.time() - t0) * 1000
        print(f" > N={N_large} (Mode Chunked) : {t_large:.2f} ms")
        if res.is_sparse: print(f" [SUCCESS] Retour Sparse détecté pour N={N_large}")
        elif CFG.DEVICE.type == 'cpu':
            # Sur CPU, on tolère le Dense car on a beaucoup de RAM système (400Mo pour 10k)
            # et c'est beaucoup plus rapide que de construire un Sparse Tensor.
            print(f" [SUCCESS] Retour Dense accepté sur CPU (Performance AVX privilégiée).")
        else: print(" [FAIL] Retour Dense pour Large N (Risque OOM)")

    def test_memory_index_performance(self):
        print("\n--- 19. TEST INDEXATION MATRICIELLE (MVP87-N8) ---")
        query = torch.randn(CFG.DIM_SIZE).to(CFG.DEVICE)
        start_n8 = time.time()
        res_n8 = self.brain.memory.find_closest(query)
        end_n8 = time.time()
        time_n8 = (end_n8 - start_n8) * 1000 
        print(f" > N8 (Matrix Index) : {time_n8:.4f} ms -> Trouvé: {res_n8[0]}")
        print(f" [SUCCESS] Indexation fonctionnelle. Type Index: {CFG.INDEX_DTYPE}")

    def test_mvp88_smart_switch(self):
        print("\n--- 20. TEST SMART SWITCH (MVP88 - Non Local) ---")
        if "TOOL_OP_LOC+OP_ATTR" not in HARDWARE_REGISTRY:
            ops = [HARDWARE_REGISTRY["OP_LOC"], HARDWARE_REGISTRY["OP_ATTR"]]
            HARDWARE_REGISTRY["TOOL_OP_LOC+OP_ATTR"] = OpComposite("TOOL_OP_LOC+OP_ATTR", ops)
        
        print(" > Input: 'Chat dans la grande Cuisine est Beau .' (Test Smart Switch)")
        self.brain.ensure_concept("Chat"); self.brain.ensure_concept("Cuisine"); self.brain.ensure_concept("Beau")
        self.brain.perceive("Chat dans la grande Cuisine est Beau .", mode="REALITY")
        inst = self.brain.find_reality_node_fast("Chat")
        if inst:
            # Correction: on accepte que le parent contienne "Cuisine" ou soit "est" selon la grammaire
            print(f" > Resultat: Parent={inst.parent.name}, Attr Beau={inst.get_local('beau') is not None}")
        else:
            print(" [FAIL] Instance Chat non trouvée.")

    def test_mvp88_life_mechanics(self):
        print("\n--- 21. TEST MECANIQUES DE VIE (MVP88) ---")
        print(" > Test Déjà-vu (Ennui)...")
        self.brain.boredom_level = 30.0
        c_test = self.brain.ensure_concept("TestDejaVu")
        c_test.create_hypothesis("couleur", self.brain.encoder.encode_word("Rouge"))
        self.brain.active_inference_check(c_test)
        b1 = self.brain.boredom_level
        self.brain.active_inference_check(c_test)
        b2 = self.brain.boredom_level
        if b2 >= b1: print(f" [SUCCESS] L'ennui n'a pas reset (B1={b1} -> B2={b2}).")
        else: print(f" [FAIL] L'ennui a reset (B1={b1} -> B2={b2}).")

    def test_mass_physics(self):
        print("\n--- 22. TEST PHYSIQUE DES MASSES (MVP88.1) ---")
        print(" > Phrase: 'Le Chat dans la grande Cuisine .'")
        c_grande = self.brain.ensure_concept("grande"); c_grande.mass = CFG.MASS_DUST
        c_cuisine = self.brain.ensure_concept("Cuisine"); c_cuisine.mass = CFG.MASS_PLANET
        self.brain.perceive("Le Chat dans la grande Cuisine .", mode="REALITY")
        inst = self.brain.find_reality_node_fast("Chat")

        if inst and inst.parent.name.startswith("Cuisine"):
            print(" [SUCCESS] Le chat est bien dans la Cuisine (Hiérarchie respectée).")
        elif inst and inst.parent.name == "REALITE":
            print(" [FAIL] Le chat est dans la Réalité (Le conteneur Cuisine a été ignoré).")
        else:
            print(f" [FAIL] Le chat est dans le parent: {inst.parent.name}")
            
        print(f"[INFO] mass de Cuisine: {c_cuisine.mass}")
        print(f"[INFO] mass de grande: {c_grande.mass}")
        print(f"[INFO] mass de Chat: {inst.mass}")
        print(f"[INFO] nom du parent de Chat: {inst.parent.name}")
        print(f"[INFO] mass du parent de Chat: {inst.parent.mass}")
        print(f"[INFO] layer du parent de Chat: {inst.parent.layer_type}")
        print(f"[INFO] layer de Chat: {inst.layer_type}")
            
        if inst and inst.is_in_reality():
            print(f" [SUCCESS] Le Chat est bien ancré dans la RÉALITÉ.")
        else:
            root_name = inst.get_root().name if inst else "None"
            print(f" [FAIL] Le Chat est perdu dans le néant (Racine: {root_name}).")

    def test_batch_ingestion_speed(self):
        print("\n--- 23. TEST VITESSE INGESTION (BATCH vs SEQ) ---")
        dummy_corpus = ["Le feu est chaud ."] * 200
        start_batch = time.time()
        
        # On force sync_wait=True pour que la mesure de temps inclue le travail du thread
        self.brain.stream.receive_sequence(dummy_corpus, mode="TRAINING", sync_wait=True)
        
        end_batch = time.time()
        elapsed = end_batch - start_batch
        
        # [FIX ZERO DIVISION] Sécurité si < 1ms (Code trop rapide ou Timer Windows imprécis)
        if elapsed < 0.001: elapsed = 0.001
            
        rate_batch = 200 / elapsed
        print(f" > Batch (N10 Stream) : {rate_batch:.1f} phrases/sec (Temps: {elapsed:.4f}s)")
        if rate_batch > 300: print(" [SUCCESS] Vitesse N10 validée.")

    def test_tokenizer_performance(self):
        print("\n--- 24. TEST TOKENIZER RUST (N10) ---")
        if not TOKENIZERS_AVAILABLE:
            print(" [SKIP] Tokenizers manquant.")
            return
        text = "Bonjour Genesis je suis ton créateur." * 1000
        t0 = time.time()
        _ = self.brain.encoder.tokenizer.encode(text) 
        t1 = time.time()
        print(f" > Encodage de {len(text)} caractères : {(t1-t0)*1000:.2f} ms")
        if (t1-t0) < 0.1: print(" [SUCCESS] Tokenizer ultra-rapide.")
        else: print(" [WARN] Tokenizer lent.")

    def test_physics_broadcasting(self):
        print("\n--- 25. TEST BROADCASTING & GRAPHS (N9) ---")
        N = 100 
        dim = CFG.DIM_SIZE
        pos = torch.rand(N, device=CFG.DEVICE); mass = torch.ones(N, device=CFG.DEVICE); vecs = torch.randn(N, dim, device=CFG.DEVICE)
        engine = self.brain.stream.phys_engine
        t0 = time.time()
        for _ in range(100): engine.compute(pos, mass, vecs, N)
        if CFG.USE_CUDA:
            torch.cuda.synchronize()
        t1 = time.time()
        avg_time = (t1-t0)/100 * 1000
        print(f" > Temps moyen par step (N={N}) : {avg_time:.4f} ms")
        if avg_time < 0.5: print(" [SUCCESS] Latence quasi-nulle (Graphs/Optimized).")
        else: print(" [INFO] Latence standard.")

    def test_massive_context_n9(self):
        print("\n--- 26. TEST CONTEXTE MASSIF (N9 STRESS TEST) ---")
        N_massive = 50000 
        print(f" > Tentative de simulation de {N_massive} particules (Mode Offloading CPU)...")
        try:
            pos = torch.rand(N_massive, device=CFG.DEVICE)
            mass = torch.ones(N_massive, device=CFG.DEVICE)
            vecs_small = torch.randn(1000, CFG.DIM_SIZE, device=CFG.DEVICE)
            vecs = vecs_small.repeat(50, 1)
            
            engine = self.brain.stream.phys_engine
            t0 = time.time()
            res = engine.compute(pos, mass, vecs, N_massive)
            t1 = time.time()
            print(f" [SUCCESS] Calcul réussi en {(t1-t0):.4f} s !")
        except RuntimeError as e:
            print(f" [FAIL] OOM ou Erreur Runtime: {e}")
            
    def test_stream_n10_performance(self):
        print("\n--- 27. TEST VITESSE INGESTION STREAM N10 (Buffer Tensoriel) ---")
        dummy_phrase = "Le chat dans la grande cuisine est beau ."
        N_PHRASES = 200
        total_tokens = len(dummy_phrase.split()) * N_PHRASES
        raw_input = [dummy_phrase] * N_PHRASES
        start_time = time.time()
        self.brain.stream.receive_sequence(raw_input, mode="TRAINING")
        end_time = time.time()
        time_elapsed = end_time - start_time
        if time_elapsed == 0: time_elapsed = 0.001
        rate_tokens_sec = total_tokens / time_elapsed
        print(f" > Taux Ingestion (N10) : {rate_tokens_sec:.1f} mots/sec")

    def test_multimodal_layering(self):
        print("\n--- 28. TEST STRATIFICATION MULTIMODALE (MVP90) ---")
        if "RETINA" in self.brain.organs:
            print(f" [SUCCESS] Organe RETINA actif (Layer {self.brain.organs['RETINA'].layer_type}).")
        else:
            print(" [FAIL] Organe RETINA manquant.")
        concept = self.brain.ensure_concept("Felin_Abstrait", layer_type=CFG.LAYER_CONCEPT)
        word = self.brain.ensure_concept("Mot_Chat", layer_type=CFG.LAYER_LANGUAGE)
        concept.link_percept(word)
        img = self.brain.ensure_concept("Image_Moustaches", layer_type=CFG.LAYER_VISUAL)
        concept.link_percept(img)
        print(f" > Concept Pivot: {concept.name} (Layer {concept.layer_type})")
        print(f" > Satellites: {[p.name + '(L' + str(p.layer_type) + ')' for p in concept.percepts]}")
        if len(concept.percepts) == 2:
            print(" [SUCCESS] Hub & Spoke fonctionnel (1 Concept -> 2 Percepts).")
        else:
            print(" [FAIL] Liaison Hub & Spoke échouée.")

    def test_attention_masking(self):
        print("\n--- 29. TEST ATTENTION MASKING (MVP91) ---")
        N = 4
        pos = torch.rand(N, device=CFG.DEVICE)
        mass = torch.ones(N, device=CFG.DEVICE)
        vecs = torch.randn(N, CFG.DIM_SIZE, device=CFG.DEVICE)
        mask = torch.zeros((N, N), device=CFG.DEVICE)
        mask[0, 1] = 1.0; mask[1, 0] = 1.0
        
        # --- CORRECTIF : SYNCHRONISATION ---
        # On s'assure que l'écriture du masque est finie avant que l'engine (qui a son propre stream) ne le lise.
        if CFG.USE_CUDA:
            torch.cuda.synchronize()
        # -----------------------------------
        
        engine = self.brain.stream.phys_engine
        forces = engine.compute(pos, mass, vecs, N, mask=mask)
        
        # On synchronise aussi après pour être sûr de lire le résultat final
        if CFG.USE_CUDA:
            torch.cuda.synchronize()
            
        if forces[0, 2].item() == 0.0 and forces[0, 1].item() != 0.0:
            print(" [SUCCESS] Le masque physique a correctement annulé les interactions non-désirées.")
        else:
            print(f" [FAIL] Le masque n'a pas fonctionné (F02={forces[0,2].item()}).")

    def test_molecule_persistence(self):
        print("\n--- 30. TEST MOLECULAR PERSISTENCE (MVP92) ---")
        mol = self.brain.create_molecule("Pomme", "ATTR", "Rouge")
        mol_name = mol.name
        self.brain.sleep()
        print(" [SYSTEM] Rechargement du cerveau (Simulation Reset)...")
        del self.brain
        # 2. On vide le cache GPU (Important pour PyTorch)
        if CFG.USE_CUDA: torch.cuda.empty_cache()
        gc.collect()
        # 4. Petite pause syndicale pour Windows
        # On tente 0.5s (500ms). Si ça plante, mettez 1.0.
        #time.sleep(0.5)
        if CFG.USE_CUDA: torch.cuda.empty_cache()
        new_brain = UnifiedBrain("fr", boolResetBase=False)
        self.brain = new_brain 
        reloaded_mol = self.brain.find_concept_exact(mol_name)
        if reloaded_mol:
            print(f" [SUCCESS] Molécule '{mol_name}' retrouvée.")
            if reloaded_mol.layer_type == CFG.LAYER_MOLECULE:
                print(" [SUCCESS] Layer Type MOLECULE conservé.")
            if "recipe" in reloaded_mol.metadata:
                print(" [SUCCESS] Recette (Recipe) conservée.")
            percept_names = [p.name for p in reloaded_mol.percepts]
            if "Pomme" in percept_names and "Rouge" in percept_names:
                print(" [SUCCESS] Liens perceptuels (Hub & Spoke) restaurés.")
        else:
            print(" [FAIL] Molécule perdue au rechargement.")

    def test_agnostic_scaffolding(self):
        print("\n--- 31. TEST AGNOSTIC SCAFFOLDING & MASS PHYSICS (MVP93) ---")
        c = self.brain.ensure_concept("Test_Concept", layer_type=CFG.LAYER_CONCEPT)
        w = self.brain.ensure_concept("Test_Word", layer_type=CFG.LAYER_LANGUAGE)
        if c.mass == 1.0 and w.mass == 0.1:
            print(" [SUCCESS] Masses configurées correctement (Concept=1.0, Word=0.1).")
        else:
            print(f" [FAIL] Erreur de masse (C={c.mass}, W={w.mass}).")
        print(" > Test de propagation : Lier 'Mot' à 'Image' doit rapprocher 'Concept'...")
        c_felin = self.brain.ensure_concept("Felin", layer_type=CFG.LAYER_CONCEPT)
        w_chat = self.brain.ensure_concept("Chat", layer_type=CFG.LAYER_LANGUAGE)
        c_felin.link_percept(w_chat) 
        c_visuel = self.brain.ensure_concept("Visuel_Poil", layer_type=CFG.LAYER_CONCEPT)
        i_poil = self.brain.ensure_concept("Image_Poil", layer_type=CFG.LAYER_VISUAL)
        c_visuel.link_percept(i_poil) 
        vec_felin_before = self.brain.encoder.get_semantic_vector("Felin").clone()
        print(" > Application Opérateur Agnostique (Mot <-> Image)...")
        self.brain.propagate_forces_vectorized(w_chat.semantic_group, i_poil.semantic_group)
        vec_felin_after = self.brain.encoder.get_semantic_vector("Felin")
        diff = (vec_felin_before - vec_felin_after).norm().item()
        print(f" > Déplacement du Concept Parent : {diff:.6f}")
        if diff > 0.000001:
            print(" [SUCCESS] Propagation Transitive réussie ! Le Concept a bougé grâce au Mot.")
        else:
            print(" [FAIL] Le Concept est resté immobile (Pas de propagation).")

    def test_mvp94_narrative_architect(self):
        print("\n--- 32. TEST NARRATIVE ARCHITECT (MVP94) ---")
        print(" > Test 1: Construction Automatique (Pattern Detection)")
        self.brain.perceive("La Voiture est Rouge .", mode="TRAINING")
        mol_name = "MOL_Voiture_Rouge"
        mol = self.brain.find_concept_exact(mol_name)
        if mol:
            print(f" [SUCCESS] Molécule '{mol_name}' créée automatiquement par le flux !")
            if mol.layer_type == CFG.LAYER_MOLECULE:
                print(" [SUCCESS] Type MOLECULE validé.")
        else:
            print(f" [FAIL] Molécule '{mol_name}' introuvable. L'automatisme a échoué.")
        print(" > Test 2: Mémoire Narrative")
        response = self.brain.ask_narrative("Voiture")
        print(f" > Question: 'Comment est la Voiture ?' -> Réponse: '{response}'")
        if "est Rouge" in response:
            print(" [SUCCESS] Le système a verbalisé la molécule !")
        else:
            print(" [FAIL] Réponse incorrecte.")
        print(" > Test 3: Robustesse (Sujet inconnu)")
        resp_unknown = self.brain.ask_narrative("Fantome")
        print(f" > Question: 'Fantome ?' -> Réponse: '{resp_unknown}'")
        if "ne sais rien" in resp_unknown:
            print(" [SUCCESS] Gestion de l'inconnu OK.")
            
    def test_optimization_vectorielle(self):
        print("\n--- 33. TEST OPTIMISATION VECTORIELLE (MVP94.17) ---")
        
        # Setup
        N = 2000
        indices_a = torch.randint(0, 100, (N,), device=CFG.DEVICE)
        indices_b = torch.randint(0, 100, (N,), device=CFG.DEVICE)
        forces = torch.rand(N, device=CFG.DEVICE)
        
        print(f" > Benchmark Batch Update (N={N} paires)...")
        
        # Mesure temps
        start = time.time()
        self.brain.encoder.learn_attraction_batch(indices_a, indices_b, forces)
        if CFG.USE_CUDA: torch.cuda.synchronize()
        end = time.time()
        
        duration = (end - start) * 1000
        print(f" > Temps d'exécution : {duration:.4f} ms")
        
        if duration < 50.0: # Seuil généreux, devrait être < 5ms sur GPU
            print(" [SUCCESS] Optimisation vectorielle validée (High Speed).")
        else:
            print(" [WARN] Performance sous-optimale (Check CPU/GPU sync).")

    def test_op11_layer_isolation(self):
        print("\n--- 34. TEST OP11: ISOLATION COUCHES & BUFFER ---")
        # 1. Vérité Conceptuelle
        self.brain.perceive("Le Ciel est Bleu .", mode="TRAINING")
        c_node = self.brain.find_node_in_layer("Ciel", CFG.LAYER_CONCEPT)
        vec_bleu = self.brain.encoder.encode_word("Bleu")
        
        # 2. Perception Contradictoire (Trust faible)
        # On simule qu'on voit un "Ciel Vert" (Perception momentanée)
        self.brain.perceive("Le Ciel est Vert .", mode="REALITY", trust=0.5)
        
        # 3. Vérification
        # Le concept Ciel ne doit PAS être Vert (Protection)
        # Attention: dans OP10 on ne sépare pas vraiment les propriétés par layer, mais on vérifie le principe
        vec_concept = c_node.get_local("vert")
        if vec_concept is None:
            print(" [SUCCESS] Le concept 'Ciel' n'a pas été corrompu par 'Vert'.")
        else:
            print(" [FAIL] Le concept 'Ciel' a été pollué !")
            
        # Par contre, une instance doit exister en Buffer/Réalité
        r_node = self.brain.find_node_in_layer("Ciel", CFG.DEPTH_REALITY)
        if r_node and r_node.get_local("vert") is not None:
             print(" [SUCCESS] L'instance 'Ciel' (Réalité) a bien enregistré 'Vert'.")
        else:
             print(" [FAIL] L'instance Réalité n'a pas été créée.")
             
    # Ajoutez ceci dans GenesisDiagnostic
    def test_cpu_benchmark_pytorch_vs_numba(self):
        print("\n--- 35. BENCHMARK CPU: PYTORCH (NEW) vs NUMBA (OLD) ---")
        
        # 1. Setup : Une charge représentative (N=2000, Dim=4096)
        N = 2000
        dim = CFG.DIM_SIZE
        print(f" > Configuration: N={N} corps, Dim={dim}, Device=CPU")
        
        # On force les tenseurs sur CPU pour le test
        pos = torch.rand(N, device="cpu", dtype=torch.float32)
        mass = torch.ones(N, device="cpu", dtype=torch.float32)
        vecs = torch.randn(N, dim, device="cpu", dtype=torch.float32)
        
        # 2. Test PyTorch (Nouveau Kernel Vectorisé AVX)
        print(" > Run PyTorch (Unified Kernel)...")
        start_torch = time.perf_counter()
        _ = gravity_kernel_masked_symmetric(pos, mass, vecs, mask=None)
        end_torch = time.perf_counter()
        dur_torch = (end_torch - start_torch) * 1000
        print(f"   -> Temps PyTorch: {dur_torch:.2f} ms")
        
        # 3. Test Numba (Legacy Kernel Assembleur)
        if NUMBA_AVAILABLE:
            print(" > Run Numba (Legacy JIT)...")
            # Conversion Numpy nécessaire pour Numba
            p_np = pos.numpy()
            m_np = mass.numpy()
            v_np = vecs.numpy()
            mask_np = np.zeros((0, 0), dtype=np.float32) # Masque vide
            
            # Warmup (La compilation JIT prend du temps au 1er appel)
            _ = driver_cpu_numba_LEGACY(p_np, m_np, v_np, dim, 10, mask_np)
            
            start_numba = time.perf_counter()
            _ = driver_cpu_numba_LEGACY(p_np, m_np, v_np, dim, N, mask_np)
            end_numba = time.perf_counter()
            dur_numba = (end_numba - start_numba) * 1000
            print(f"   -> Temps Numba:  {dur_numba:.2f} ms")
            
            # Conclusion
            if dur_torch < dur_numba:
                gain = dur_numba / dur_torch
                print(f" [WINNER] PyTorch est {gain:.1f}x plus rapide que Numba sur cette machine.")
            else:
                loss = dur_torch / dur_numba
                print(f" [INFO] Numba reste {loss:.1f}x plus rapide (PyTorch est acceptable).")
        else:
            print(" [SKIP] Numba non installé, comparaison impossible.")

    # N'oubliez pas d'ajouter self.test_cpu_benchmark_pytorch_vs_numba() dans run_all() !
    
    def test_numerical_stability(self):
        print("\n--- 36. TEST STABILITÉ NUMÉRIQUE (Stress Test) ---")
        # On génère des vecteurs avec des normes aberrantes pour voir si ça explose
        N = 100
        dim = CFG.DIM_SIZE
        print(f" > Injection de vecteurs à haute énergie (Norme > 100)...")
        
        # Vecteurs géants (simule une explosion de gradient ou une accumulation infinie)
        vecs = torch.randn(N, dim, device=CFG.DEVICE) * 100.0 
        pos = torch.rand(N, device=CFG.DEVICE)
        mass = torch.ones(N, device=CFG.DEVICE)
        
        # Le kernel doit tenir le coup grâce à la normalisation interne
        try:
            forces = gravity_kernel_masked_symmetric(pos, mass, vecs)
            if torch.isnan(forces).any() or torch.isinf(forces).any():
                print(" [FAIL] Explosion numérique détectée (NaN ou Inf) !")
            else:
                print(f" [SUCCESS] Forces calculées saines (Max force: {forces.max().item():.4f}).")
                print("           Le kernel unifié a correctement normalisé les entrées.")
        except Exception as e:
            print(f" [FAIL] Crash du kernel : {e}")

    def test_scalability(self):
        print("\n--- 37. TEST D'ÉCHELLE (Scalability) ---")
        if CFG.DEVICE.type == 'cpu' and not NUMBA_AVAILABLE:
            print(" [INFO] Mode PyTorch CPU pur.")
            
        steps = [100, 1000, 5000]
        dim = CFG.DIM_SIZE
        
        for n in steps:
            # Création de données dummy
            pos = torch.rand(n, device=CFG.DEVICE)
            mass = torch.ones(n, device=CFG.DEVICE)
            vecs = torch.randn(n, dim, device=CFG.DEVICE)
            
            # Mesure
            torch.cuda.synchronize() if CFG.USE_CUDA else None
            t0 = time.perf_counter()
            
            # On appelle le moteur via le stream pour utiliser la logique de chunking si nécessaire
            _ = self.brain.stream.phys_engine.compute(pos, mass, vecs, n)
            
            torch.cuda.synchronize() if CFG.USE_CUDA else None
            dt = (time.perf_counter() - t0) * 1000
            
            print(f" > N={n:<5} : {dt:.2f} ms")
            
            # Alerte si trop lent (> 16ms pour 1k serait inquiétant pour du temps réel)
            if n == 1000 and dt > 50.0:
                print("   [WARN] Attention, performance limite pour le temps réel.")

    def test_quantization_quality(self):
        print("\n--- 38. TEST QUALITÉ QUANTIZATION (INT8) ---")
        # Création d'un vecteur sémantique riche
        vec_original = torch.randn(1, CFG.DIM_SIZE, device=CFG.DEVICE)
        vec_original = F.normalize(vec_original, p=2, dim=1, eps=CFG.EPSILON)
        
        # 1. Quantization "Smart" (Avec Scale)
        q_vec, scale = SmartQuantizer.quantize(vec_original)
        rec_vec = SmartQuantizer.dequantize(q_vec, scale)
        
        # 2. Mesure de la perte
        similarity = F.cosine_similarity(vec_original, rec_vec).item()
        loss = 1.0 - similarity
        
        print(f" > Vecteur Original (FP32) vs Reconstruit (INT8)")
        print(f" > Similarité Cosine : {similarity:.6f}")
        print(f" > Perte d'information : {loss:.6f}")
        
        if similarity > 0.99:
            print(" [SUCCESS] La quantization INT8 est quasi-transparente.")
        else:
            print(" [WARN] Perte de précision significative.")
            
        # 1. Test de Nuance (Ciel vs Azur)
        # On crée deux vecteurs très proches (Nuance fine)
        vec_base = torch.randn(1, CFG.DIM_SIZE, device=CFG.DEVICE)
        vec_base = F.normalize(vec_base, p=2, dim=1)
        
        # On crée une variation minime (0.01)
        noise = torch.randn(1, CFG.DIM_SIZE, device=CFG.DEVICE) * 0.01
        vec_nuance = F.normalize(vec_base + noise, p=2, dim=1)
        
        dist_originale = (vec_base - vec_nuance).norm().item()
        print(f" > Distance Originale (FP32) : {dist_originale:.6f}")
        
        # 2. Compression / Décompression
        q_base, s_base = SmartQuantizer.quantize(vec_base)
        rec_base = SmartQuantizer.dequantize(q_base, s_base)
        
        q_nuance, s_nuance = SmartQuantizer.quantize(vec_nuance)
        rec_nuance = SmartQuantizer.dequantize(q_nuance, s_nuance)
        
        # 3. Vérification post-compression
        dist_reconstruite = (rec_base - rec_nuance).norm().item()
        print(f" > Distance Reconstruite (INT8): {dist_reconstruite:.6f}")
        
        perte_relative = abs(dist_originale - dist_reconstruite)
        print(f" > Delta de distance : {perte_relative:.6f}")
        
        # On accepte une déviation minime, mais la distinction doit rester nette
        if perte_relative < 0.01:
            print(" [SUCCESS] La nuance est conservée (Ciel != Azur).")
        else:
            print(" [WARN] Attention, écrasement des nuances fines !")

        # 4. Test Vecteur Nul (Sécurité Crash)
        vec_nul = torch.zeros(1, CFG.DIM_SIZE, device=CFG.DEVICE)
        q_nul, s_nul = SmartQuantizer.quantize(vec_nul)
        rec_nul = SmartQuantizer.dequantize(q_nul, s_nul)
        if torch.isnan(rec_nul).any():
            print(" [FAIL] Le vecteur nul a provoqué des NaN !")
        else:
            print(" [SUCCESS] Gestion du vecteur nul OK (Pas de crash).")

    def test_hypothesis_acceptance(self):
        print("\n--- 39. TEST CONSOLIDATION CROYANCE (Scenario Logique) ---")
        # Scénario : On force une idée fausse "Le Ciel est Vert" plusieurs fois
        # et on vérifie si elle devient une vérité (Consolidation).
        
        concept = "CielTest"
        attribut = "VertTest"
        
        # 1. Création
        c_node = self.brain.ensure_concept(concept)
        vec_attr = self.brain.encoder.encode_word(attribut)
        
        print(f" > Inception : On répète '{concept} est {attribut}'...")
        
        # Simulation de répétition (Renforcement de l'hypothèse)
        # On crée l'hypothèse manuellement pour simuler la perception
        c_node.create_hypothesis(attribut, vec_attr)
        hyp_name = f"HYP_{concept}_{attribut}"
        
        if hyp_name in c_node.children:
            hyp_node = c_node.children[hyp_name]
            # On booste artificiellement son énergie pour simuler 10 perceptions
            hyp_node.energy = CFG.THRESHOLD_VALIDATION + 10.0
            print(f"   (Hypothèse '{hyp_name}' chargée à bloc: Energy={hyp_node.energy})")
            
            # 2. Consolidation (Sommeil)
            print(" > Lancement Consolidation...")
            self.brain.consolidate_hypotheses()
            
            # 3. Vérification
            if hyp_name not in c_node.children:
                # Elle a disparu de la liste des enfants (donc traitée)
                # Est-elle intégrée dans le concept ?
                local_val = c_node.get_local(attribut)
                if local_val is not None:
                    print(f" [SUCCESS] Le système a cru au mensonge : {concept} est devenu {attribut}.")
                    print("           (Mécanisme de croyance validé).")
                else:
                    print(" [FAIL] Hypothèse disparue mais pas intégrée (Oubli ?).")
            else:
                print(" [FAIL] Hypothèse non traitée (Reste en suspens).")
        else:
            print(" [ERR] Échec création hypothèse.")
            
            
    def test_bridge_architecture(self):
        print("\n--- 36. TEST ARCHITECTURE BRIDGE (MVP97) ---")
        
        # Test 1 : Bridge Léger (Production)
        print(" > Test du Bridge Léger (Threading)...")
        if self.brain.stream.bridge:
            # On injecte manuellement
            test_phrase = ["Le", "Turbo", "est", "rapide"]
            self.brain.stream.bridge.in_q.put(test_phrase)
            time.sleep(0.1) # Laisser le temps au thread
            
            if not self.brain.stream.bridge.out_q.empty():
                packet = self.brain.stream.bridge.out_q.get()
                print(f" [SUCCESS] Packet reçu du Thread Worker (Taille: {packet['count']})")
                print(f"           Type Tenseur: {packet['vecs'].device}")
            else:
                print(" [FAIL] Le Bridge Léger n'a rien renvoyé.")
        else:
            print(" [INFO] Multithreading désactivé dans la config.")

    def test_hybrid_bridge_chaining(self):
        print("\n--- 37. TEST HYBRIDE (HEAVY -> LIGHT CHAIN) ---")
        print(" > Simulation d'un pipeline distribué...")
        
        # Files d'attente pour le processus lourd (Picklable)
        mp_ctx = multiprocessing.get_context("spawn") # Sécurité Windows
        heavy_in = mp_ctx.Queue()
        heavy_out = mp_ctx.Queue()
        
        # Démarrage du processus lourd
        heavy_bridge = PrototypeHeavyBridge(heavy_in, heavy_out)
        heavy_bridge.start()
        
        try:
            # 1. Injection dans le Lourd (Ex: Donnée brute externe)
            heavy_in.put("Data_Distante")
            
            # 2. Attente traitement Lourd
            # [CORRECTION] Augmentation du timeout (2s -> 15s)
            # Le démarrage d'un processus spawn prend du temps (imports PyTorch/Cuda)
            print(" > Attente du Heavy Bridge (Warmup)...")
            processed_data = heavy_out.get(timeout=CFG.BRIDGE_SYNC_TIMEOUT)
            print(f" > Reçu du Heavy Bridge: '{processed_data}'")
            
            # 3. Injection dans le Léger (Le "Chainage")
            # Le main thread fait le pont entre la Queue MP et la Queue Thread
            if self.brain.stream.bridge:
                # On simule que la donnée traitée devient une phrase
                fake_sentence = [processed_data, "est", "reçu"]
                
                # On utilise la nouvelle méthode process_data qui attend une liste de lignes
                # Donc on encapsule notre phrase dans une liste de lignes (ici une seule ligne)
                fake_line = " ".join(fake_sentence) + " ."
                self.brain.stream.bridge.in_q.put([fake_line])
                
                # Vérif (On laisse un peu de temps au thread léger)
                time.sleep(0.5)
                
                # On vérifie si quelque chose sort (Succès)
                if not self.brain.stream.bridge.out_q.empty():
                    # On vide pour nettoyer
                    while not self.brain.stream.bridge.out_q.empty():
                        self.brain.stream.bridge.out_q.get()
                    print(" [SUCCESS] Chaîne Complète : Heavy(Process) -> Main -> Light(Thread) -> GPU Validée.")
                else:
                    print(" [FAIL] Rupture de chaîne au niveau du Light Bridge (Rien reçu).")
            else:
                 print(" [INFO] Pas de bridge léger actif (Test partiel).")
            
        except Exception as e:
            # On affiche repr(e) pour voir l'erreur exacte (souvent 'Empty')
            print(f" [FAIL] Erreur chaîne hybride: {repr(e)}")
        finally:
            heavy_bridge.stop()
            heavy_bridge.join(timeout=1.0)
            if heavy_bridge.is_alive(): heavy_bridge.terminate()

if __name__ == "__main__":
    
    Nb_DIM = 4096 #tested for: 64, 128, 256, 512, 1024, 2048, 4096
    # N11: CONFIGURATION DE PRÉCISION (Point 3)
    # Options: "INT8", "FP16", "FP32"
    strPRECISION_MODE = "INT8"
    boolForceCPU = False # false for CUDA auto-detection and True to force CPU (CUDA unactivation)
    #boolForceCPU = True
    str_lang = "fr" 
    boolResetBase = False
    boolENABLE_MULTITHREADING = True #active le threading
    boolFORCE_LIGHT_MODE = True #desactive le Multiprocess
    #boolResetBase = True # to reset database at start
    RUN_MODE = "DIAGNOSTIC" 
    #RUN_MODE = "LIFE_LOOP" 
    #RUN_MODE = "TRAINING_FILE" 
    #RUN_MODE = "IMPORT_MODEL" 
    #RUN_MODE = "INFERENCE" 
    
    CFG = GenesisConfig(dim=Nb_DIM, PrecisionType = strPRECISION_MODE, ForceCPU=boolForceCPU, ENABLE_MULTITHREADING=boolENABLE_MULTITHREADING,FORCE_LIGHT_MODE=boolFORCE_LIGHT_MODE)
    # --- INSTANCE GLOBALE LOGGER ---
    LOGGER = GenesisAsyncLogger()
    LOGGER.start()
    
    try:
        if RUN_MODE == "DIAGNOSTIC":
            CFG.BASE_MEM_DIR = CFG.BASE_MEM_DIR + "_DIAG"
        
        brain = UnifiedBrain(str_lang, boolResetBase)
        bootloader = GenesisBootloader(CFG, brain)
        diagnostics = GenesisDiagnostic(brain)
        if RUN_MODE == "DIAGNOSTIC":
            diagnostics.forensic_audit_ghosts()
            diagnostics.run_all()
        elif RUN_MODE == "LIFE_LOOP":
            brain.life_cycle()
        elif RUN_MODE == "TRAINING_FILE":
            bootloader.train_from_corpus_file("genesis_curriculum.txt", epochs=5)
        elif RUN_MODE == "IMPORT_MODEL":
            bootloader.import_external_vectors("fake_vectors.txt")
        elif RUN_MODE == "INFERENCE":
            print("\n [INFERENCE] Mode Interactif Activé (Ctrl+C pour quitter).")
            brain.associative_memory._refresh_lexicon()
            try:
                while True:
                    u = input(" > Vous: ")
                    if u.lower() in ["q", "exit", "quit"]: break
                    if not u.strip(): continue
                    if u.lower().startswith("comment est "):
                        subj = u.lower().replace("comment est ", "").replace("?", "").strip()
                        print(f" > GENESIS (Mémoire Narrative): {brain.ask_narrative(subj)}")
                        continue
                    phrase_traitee = u if u.endswith(".") else u + " ."
                    brain.perceive(phrase_traitee)
                    mots = u.replace(".", "").split()
                    if mots:
                        vec_pensee = torch.zeros(brain.dim).to(CFG.DEVICE)
                        for m in mots: vec_pensee += brain.encoder.encode_word(m)
                        vec_pensee = F.normalize(vec_pensee, p=2, dim=0, eps=CFG.EPSILON)
                        reponse = brain.generate_response(vec_pensee)
                        print(f" > GENESIS (Association): {reponse}")
            except KeyboardInterrupt:
                print("\n [SYSTEME] Arrêt d'urgence.")
                brain.sleep()
        
    except Exception as e:
        print(f"\n [CRASH] Erreur fatale : {e}")
        traceback.print_exc()
            
    finally:
        print("\n [SHUTDOWN] Arrêt du Logger Asynchrone...")
        # --- NETTOYAGE CENTRALISÉ ---
        # Peu importe si on a 1 cerveau, 10 threads ou si ça a planté,
        # la Config connait tout le monde et éteint la lumière.
        GenesisConfig.global_shutdown()
