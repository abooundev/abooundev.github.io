---
title: "Jekyll Theme 커스터마이징"
date: 2020-09-12
math: true

---

#  Jekyll Theme 커스터마이징

### favicon 바꾸기

* https://chirpy.cotes.info/posts/customize-the-favicon/



### 수식 적용 하기

* 포스팅에 수식 설정 옵션 설정하기

  * ```
    ---
    math: true
    ---
    ```

  * https://chirpy.cotes.info/posts/write-a-new-post/

* _includes/js-selector.html에 inline 수식 설정하기

  * ```html
    {% if page.math %}
        <!-- MathJax -->
        <!-- inlineMath 추가 -->
        <script>
            MathJax = {
                tex: {
                    inlineMath: [['$', '$'], ['\\(', '\\)']]
                }
            };
        </script>
        <script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
        <script id="MathJax-script" async
                src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js">
        </script>
    {% endif %}
    ```

  * https://pangtrue.tistory.com/113

  * MathJax 공식 문서 

    * http://docs.mathjax.org/en/latest/basic/mathematics.html#basic-mathematics



### 원하는 키워드 파일 위치 찾기

* grep 명렁어 사용

  * include 옵션을 사용해 특정 확장자만 검색 가능

  * ```shell
    $ grep -r '검색어' /폴더경로/* [--include '*.conf'] # include 옵션을 사용해 특정 확장자만 검색
    $ grep -r 'MathJax' ./abooundev.github.io/*  
    ```

    



### 참고

* 구조 파악
  * http://jihyeleee.com/blog/third-designer-can-make-jekyll-blog/

* 같은 테마 적용 블로그
  * https://baeseongsu.github.io/posts/apply-mathjax-to-jekyll-blog/
