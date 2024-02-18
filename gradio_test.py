from gradio_client import Client

client = Client("https://b779e04fb00b60ddae.gradio.live/")
result = client.predict(
		"https://github.com/gradio-app/gradio/raw/main/test/test_files/audio_sample.wav",	# filepath  in 'new_chunk' Audio component
		api_name="/predict"
)
print(result)