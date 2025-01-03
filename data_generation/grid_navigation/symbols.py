
obj_choices = "heart & crown & flag & star & flower & umbrella & plane & phone & spark & diamond & queen & hammer & club & gear & arrow & sun & bishop & note & coffee & anchor & cloud & pawn & castle & horse & infinity & moon & null & approx & integral & product & sum"
obs_choices = "dot & cross & square  & triangle & plus"
obj_choices = obj_choices.split("&")
obs_choices = obs_choices.split("&")

obj_choices = [" " + item.strip() for item in obj_choices]
obs_choices = [" " + item.strip() for item in obs_choices]

obj_choices_symbols = ['❤️', '♔', '⚑', '★', '✿', '☂', '✈', '☎', '✦', '❖', '♛', '⚒︎', '♣︎', '⚙', '➜', '☀', "♗", "♪", "☕", "⚓", "☁",  "♙", "♖", "♘", "∞", "☾", "∅", "≈", "∫", "∏", "∑",]
obs_choices_symbols = ['●', '✖', '◼️', '▲', '✚']


