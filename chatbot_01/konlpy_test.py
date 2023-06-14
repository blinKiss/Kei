import konlpy

# 객체 생성
okt = konlpy.tag.Okt()

word = okt.morphs('와우 오늘 봤던 영화가 흥미롭고 재미있네', stem=True)

print(word)