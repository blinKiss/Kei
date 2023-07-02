# 문제1
# KONLPY 라이브러리를 설치하고 다음 문장을 형태소로 분리하는 코드를 작성하시오 (test1.py)
 
# setence = '와 정말 영화 재밌다 뮤직비디오도 정말 좋은 것 같아!'
import konlpy

# 객체 생성
okt = konlpy.tag.Okt()

# 문제엔 setence지만 sentence로 바꿈
sentence = '와 정말 영화 재밌다 뮤직비디오도 정말 좋은 것 같아!'

sentence_split = okt.morphs(sentence)

print(sentence_split)