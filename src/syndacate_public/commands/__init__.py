from juml.base import Command

from syndacate_public.commands.plot_part_predictions import PlotPartPredictions
from syndacate_public.commands.plot_char_predictions import PlotCharPredictions
from syndacate_public.commands.plot_word_predictions import PlotWordPredictions
from syndacate_public.commands.plot_syndacate import PlotSyndacate

def get_all() -> list[type[Command]]:
    return [
        PlotPartPredictions,
        PlotCharPredictions,
        PlotWordPredictions,
        PlotSyndacate,
    ]
