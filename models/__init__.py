from .static_edge import Net as StaticEdgeNet
from .gat         import Net as GATNet
from .transformer import Net as TransformerNet
from .secgat      import Net as SECGATNet

MODELS = {
    'static_edge': StaticEdgeNet,
    'gat':         GATNet,
    'transformer': TransformerNet,
    'secgat':      SECGATNet,
}
