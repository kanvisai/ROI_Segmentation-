from types import SimpleNamespace

from detector_palma.etiquetas import (
    categoria_de_etiqueta,
    es_clase_nombrada,
    heredar_etiqueta,
    mezclar_votos_clase,
    resumir_deteccion_objeto,
)


def test_mapea_coco_a_catalogo():
    assert categoria_de_etiqueta("bottle") == "botella"
    assert categoria_de_etiqueta("Cell Phone") == "celular"
    assert categoria_de_etiqueta("cup") == "vaso"
    assert categoria_de_etiqueta("objeto") == "objeto"
    assert categoria_de_etiqueta("sports ball") == "sports_ball"
    assert not es_clase_nombrada("objeto")
    assert es_clase_nombrada("bottle")


def test_resumir_elige_clase_con_mejor_score():
    cat, conf, raw = resumir_deteccion_objeto(
        {0: "objeto", 1: "bottle", 2: "bottle", 3: "cup"},
        {0: 1.0, 1: 0.41, 2: 0.88, 3: 0.90},
    )
    assert (cat, conf, raw) == ("botella", 0.88, "bottle")


def test_resumir_sin_clase_yolo():
    cat, conf, raw = resumir_deteccion_objeto(
        {0: "objeto", 1: "object"},
        {0: 1.0, 1: 0.999},
    )
    assert cat == "objeto"
    assert conf is None
    assert raw == "objeto"


def test_resumir_sin_clase_usa_score_real():
    cat, conf, raw = resumir_deteccion_objeto(
        {0: "objeto", 1: "objeto"},
        {0: 0.44, 1: 0.61},
    )
    assert (cat, conf, raw) == ("objeto", 0.61, "objeto")


def test_resumir_sin_clase_usa_agarre_si_el_score_es_falso():
    cat, conf, raw = resumir_deteccion_objeto(
        {0: "objeto", 1: "objeto"},
        {0: 1.0, 1: 0.999},
        fallback_scores=[0.41, 0.72, 0.0],
    )
    assert (cat, conf, raw) == ("objeto", 0.72, "objeto")


def test_heredar_etiqueta_sam_desde_yolo():
    sam = SimpleNamespace(etiqueta="objeto", score=1.0, extra={})
    yolo = SimpleNamespace(etiqueta="bottle", score=0.73, extra={"clase_id": 39})
    heredar_etiqueta(sam, yolo)
    assert sam.etiqueta == "bottle"
    assert sam.score == 0.73
    assert sam.extra["clase_id"] == 39
    heredar_etiqueta(yolo, SimpleNamespace(etiqueta="cup", score=0.20, extra={}))
    assert yolo.etiqueta == "bottle"


def test_mezclar_votos_conserva_la_mejor_clase():
    etiq, sc = mezclar_votos_clase(
        {1: "objeto"},
        {1: 0.0},
        {1: "cell phone", 2: "bottle"},
        {1: 0.61, 2: 0.40},
    )
    assert etiq[1] == "cell phone"
    assert sc[1] == 0.61
    cat, conf, raw = resumir_deteccion_objeto(etiq, sc)
    assert cat == "celular"
    assert conf == 0.61
    assert raw == "cell phone"
