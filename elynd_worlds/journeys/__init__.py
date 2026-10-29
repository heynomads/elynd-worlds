"""Journeys: a 30-day goal, one meditation for each of its days.

A listener tells a guide what they would like to grow in the next 30 days (`guide`); a plan for the thirty days is
made from it (`planner`); every day the meditation's writer gets that day's part of the journey, so the sessions
remember each other (`brief`): day one opens the journey, a session picks up from the last one and from how the
listener says it went, every seventh day looks back on the week, and the thirtieth closes the circle.

Whose model reads what. Everything that reads a listener's own words (the conversation, the plan, the day's block)
runs on Meta's open-weights Llama 3.3 70B inside ELYND's own cloud (Amazon Bedrock, `llama`): Meta's processor terms
for its hosted API (Meta Global Processor Terms 2.2) bar health-related data, and a goal and its notes can carry it,
so the listener's words never reach Meta. (Llama 4 is multimodal, and its licence grants no rights to companies based
in the European Union; ELYND is one.) Only the non-personal description of the journey's world goes to Meta's API, to
paint it with Muse Image and check it with Muse Spark Open, as every world is.

The journey's world grows with it (`stages`): five living worlds of one place, the first painted, the other four
repainted from it in another light and season, from the same viewpoint.
"""
