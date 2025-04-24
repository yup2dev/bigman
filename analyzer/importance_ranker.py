from sklearn.feature_extraction.text import TfidfVectorizer

def calculate_tfidf_importance(articles, keyword_list):
    # 텍스트를 소문자로 변환하여 대소문자 구분 제거
    texts = [article['text'].lower() for article in articles]

    # TF-IDF 행렬 생성
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(texts)

    # 키워드를 소문자로 변환
    keyword_list = [keyword.lower() for keyword in keyword_list]

    # 중요도 계산
    keyword_importance = []
    seen_urls = set()  # 중복 기사 제거를 위한 URL 저장
    for i, article in enumerate(articles):
        # 중복 기사 체크 (URL이 있는 경우)
        article_url = article.get('url', '')
        if article_url in seen_urls:
            continue
        seen_urls.add(article_url)

        # TF-IDF 점수 합산
        score = 0
        for keyword in keyword_list:
            if keyword in vectorizer.vocabulary_:  # 키워드가 어휘 사전에 있는 경우
                keyword_index = vectorizer.vocabulary_[keyword]
                score += tfidf_matrix[i, keyword_index]  # 해당 키워드의 TF-IDF 점수 추가

        # 중요도 점수 저장
        article['importance_score'] = score
        if score > 0:  # 중요도 점수가 0보다 큰 기사만 추가
            keyword_importance.append((article, score))

    keyword_importance.sort(key=lambda x: x[1], reverse=True)

    # 정렬된 기사 리스트 반환
    return [article for article, _ in keyword_importance]