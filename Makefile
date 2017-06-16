all:
	@echo "done"

run:
	@echo "running prediction for digit recognizer with 50 hidden layer neurons"
	@python digit_recognizer.py digit predict 50
