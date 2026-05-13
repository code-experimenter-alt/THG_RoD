# -*- coding: utf-8 -*-

from typing import Dict, Any

PRESETS: Dict[str, Dict[str, Any]] = {
    "G0a_NonDP_cnn_vanilla": dict(
        feature_backend="mel",
        teacher_backend="mel",
        student_backend="mel",
        model_teacher="cnn_mel",
        model_student=None,
        train_teacher=True,
        dp_teacher=False,
        student_mode="hard",
        loss_name="ce",
        n_mels=40,
        max_time=100,
        batch_size=128,
        epochs_teacher=8,
        lr_teacher=1e-3,
        weight_decay=1e-4,
    ),

    "G0b_SSL_linear_nonDP": dict(
        feature_backend="ssl",
        teacher_backend="ssl",
        student_backend="ssl",
        model_teacher="linear_head",
        model_student=None,
        train_teacher=True,
        dp_teacher=False,
        student_mode="hard",
        loss_name="ce",
        ssl_model_name="facebook/wav2vec2-base",
        ssl_pooling="mean",
        ssl_max_seconds=12.0,
        batch_size=64,
        epochs_teacher=8,
        lr_teacher=1e-3,
        weight_decay=1e-4,
    ),

    "G1a_DPdirect_cnn_vanilla": dict(
        feature_backend="mel",
        teacher_backend="mel",
        student_backend="mel",
        model_teacher="cnn_mel",
        model_student=None,
        train_teacher=True,
        dp_teacher=True,
        student_mode="hard",
        loss_name="ce",
        n_mels=40,
        max_time=100,
        batch_size=128,
        epochs_teacher=8,
        lr_teacher=1e-3,
        weight_decay=1e-4,
        max_grad_norm=1.0,
        delta=1e-5,
    ),

    "G1b_DPdirect_ssl_linear": dict(
        feature_backend="ssl",
        teacher_backend="ssl",
        student_backend="ssl",
        model_teacher="linear_head",
        model_student=None,
        train_teacher=True,
        dp_teacher=True,
        student_mode="hard",
        loss_name="ce",
        ssl_model_name="facebook/wav2vec2-base",
        ssl_pooling="mean",
        ssl_max_seconds=12.0,
        batch_size=128,
        epochs_teacher=8,
        lr_teacher=1e-3,
        weight_decay=1e-4,
        max_grad_norm=1.0,
        delta=1e-5,
    ),

    # Main proposed release pipeline
    "CV_SSL_BS_RELEASE": dict(
        feature_backend="ssl",
        teacher_backend="ssl",
        student_backend="ssl",
        model_teacher="linear_head",
        model_student="linear_head",

        train_teacher=True,
        dp_teacher=True,
        student_mode="kd",
        loss_name="balanced_softmax",

        ssl_model_name="facebook/wav2vec2-base",
        ssl_pooling="mean",
        ssl_max_seconds=12.0,

        batch_size=128,
        epochs_teacher=8,
        epochs_student=15,
        lr_teacher=1e-3,
        lr_student=1e-3,
        weight_decay=1e-4,

        max_grad_norm=8.0,
        delta=1e-5,

        kd_alpha=0.7,
        kd_temperature=2.0,

        health_power=1.0,
        health_floor=0.0,
        tcrd_tau_low=0.45,
        tcrd_tau_high=0.60,
    ),

    # Weak teacher diagnostic
    "CV_MEL_CE_WEAK_RELEASE": dict(
        feature_backend="mel",
        teacher_backend="mel",
        student_backend="mel",
        model_teacher="cnn_mel",
        model_student="cnn_mel",

        train_teacher=True,
        dp_teacher=True,
        student_mode="kd",
        loss_name="ce",

        n_mels=40,
        max_time=100,
        use_dsaf=True,
        eta0=1e-5,

        batch_size=128,
        epochs_teacher=8,
        epochs_student=15,
        lr_teacher=1e-3,
        lr_student=1e-3,
        weight_decay=1e-4,

        max_grad_norm=8.0,
        delta=1e-5,

        kd_alpha=0.7,
        kd_temperature=2.0,

        health_power=1.0,
        health_floor=0.0,
        tcrd_tau_low=0.45,
        tcrd_tau_high=0.60,
    ),

    "CV_MEL_BS_WEAK_RELEASE": dict(
        feature_backend="mel",
        teacher_backend="mel",
        student_backend="mel",
        model_teacher="cnn_mel",
        model_student="cnn_mel",

        train_teacher=True,
        dp_teacher=True,
        student_mode="kd",
        loss_name="balanced_softmax",

        n_mels=40,
        max_time=100,
        use_dsaf=True,
        eta0=1e-5,

        batch_size=128,
        epochs_teacher=8,
        epochs_student=15,
        lr_teacher=1e-3,
        lr_student=1e-3,
        weight_decay=1e-4,

        max_grad_norm=8.0,
        delta=1e-5,

        kd_alpha=0.7,
        kd_temperature=2.0,

        health_power=1.0,
        health_floor=0.0,
        tcrd_tau_low=0.45,
        tcrd_tau_high=0.60,
    ),
}

MINIMAL_SET = [
    "CV_SSL_BS_RELEASE",
    "CV_MEL_CE_WEAK_RELEASE",
]