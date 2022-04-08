python main.py --model_type ngram --preprocess 1 --part 2 --N 1 --num_feature 250 > results/1_250.txt &
python main.py --model_type ngram --preprocess 1 --part 2 --N 1 --num_feature 500 > results/1_500.txt &
python main.py --model_type ngram --preprocess 1 --part 2 --N 1 --num_feature 750 > results/1_750.txt &
python main.py --model_type ngram --preprocess 1 --part 2 --N 1 --num_feature 1000 > results/1_1000.txt &
python main.py --model_type ngram --preprocess 1 --part 2 --N 1 --num_feature 1500 > results/1_1500.txt &
python main.py --model_type ngram --preprocess 1 --part 2 --N 1 --num_feature 1500 > results/1_2000.txt &
python main.py --model_type ngram --preprocess 1 --part 2 --N 1 --num_feature 1500 > results/1_2500.txt &

python main.py --model_type ngram --preprocess 1 --part 2 --N 2 --num_feature 250 > results/2_250.txt &
python main.py --model_type ngram --preprocess 1 --part 2 --N 2 --num_feature 500 > results/2_500.txt &
python main.py --model_type ngram --preprocess 1 --part 2 --N 2 --num_feature 750 > results/2_750.txt &
python main.py --model_type ngram --preprocess 1 --part 2 --N 2 --num_feature 1000 > results/2_1000.txt &
python main.py --model_type ngram --preprocess 1 --part 2 --N 2 --num_feature 1500 > results/2_1500.txt &
python main.py --model_type ngram --preprocess 1 --part 2 --N 2 --num_feature 1500 > results/2_2000.txt &
python main.py --model_type ngram --preprocess 1 --part 2 --N 2 --num_feature 1500 > results/2_2500.txt &

python main.py --model_type ngram --preprocess 1 --part 2 --N 3 --num_feature 250 > results/3_250.txt &
python main.py --model_type ngram --preprocess 1 --part 2 --N 3 --num_feature 500 > results/3_500.txt &
python main.py --model_type ngram --preprocess 1 --part 2 --N 3 --num_feature 750 > results/3_750.txt &
python main.py --model_type ngram --preprocess 1 --part 2 --N 3 --num_feature 1000 > results/3_1000.txt &
python main.py --model_type ngram --preprocess 1 --part 2 --N 3 --num_feature 1500 > results/3_1500.txt &
python main.py --model_type ngram --preprocess 1 --part 2 --N 3 --num_feature 1500 > results/3_2000.txt &
python main.py --model_type ngram --preprocess 1 --part 2 --N 3 --num_feature 1500 > results/3_2500.txt &

wait

echo "ALL DONE"