# 017_LLM-02_opneai-rrg-sample

RAG（Retrieval-Augmented Generation）をゼロから実装するサンプルプロジェクトです。

このプロジェクトは、[Zenn 記事「RAG をゼロから実装して仕組みを学ぶ【2025 年版】」](https://zenn.dev/knowledgesense/articles/2619c6e5918d08)を参考に実装されています。

## 概要

Streamlit と Faiss を使用したシンプルな RAG アプリケーションです。ローカルのテキストファイルをベクトル化し、ユーザーの質問に対して関連する情報を検索して回答を生成します。

## 技術スタック

-   **Python 3.x**
-   **Streamlit** - Web アプリフレームワーク
-   **Faiss** - ベクトル検索ライブラリ
-   **OpenAI API** - 埋め込み生成と回答生成
-   **uv** - パッケージ管理

## セットアップ

### 1. uv のインストール

macOS の場合:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Windows の場合（PowerShell を管理者として実行）:

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

インストール後、新しくターミナルを開き直して、`uv --version` で確認してください。

### 2. Python のインストール

```bash
uv python install
```

### 3. プロジェクトの初期化

```bash
uv init
```

### 4. 依存パッケージのインストール

```bash
uv add streamlit faiss-cpu python-dotenv openai numpy
```

### 5. 環境変数の設定

`.env.example` を `.env` にコピーして、OpenAI API キーを設定してください。

```bash
cp .env.example .env
```

`.env` ファイルを編集して、実際の API キーを設定します：

```
OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxx
```

**注意**:

-   OpenAI API キーの取得方法は、[OpenAI 公式サイト](https://platform.openai.com/api-keys)を参照してください
-   API キーは他人に見せず、大切に保管してください
-   この RAG を動かすことで、質問ごとに数円程度、OpenAI に対してお金がかかります（OpenAI の料金ダッシュボードを確認しましょう）

### 6. データファイルの準備

`data/` ディレクトリに `.txt` ファイルを配置してください。サンプルとして `data/knowledge.txt` が含まれています。

## 実行方法

以下のコマンドでアプリケーションを起動します：

```bash
uv run streamlit run app.py
```

自動的に `http://localhost:8501/` が開かれ、RAG アプリケーションが利用できるようになります。

## 使い方

1. ブラウザでアプリケーションが開いたら、質問を入力してください
2. 「RAG に聞いてみる」ボタンをクリックします
3. 検索されたコンテキストと生成された回答が表示されます

### 設定

サイドバーから以下の設定が可能です：

-   **参照するドキュメント数**: 検索結果として表示するドキュメントの数を調整できます（1〜10 件）

## プロジェクト構造

```
uik/
├── app.py              # メインアプリケーション
├── data/
│   └── knowledge.txt   # サンプル知識ベース
├── .env.example        # 環境変数テンプレート
├── .gitignore          # Git除外設定
├── pyproject.toml      # uvプロジェクト設定
└── README.md           # このファイル
```

## 機能

-   **ドキュメント読み込み**: `data/` ディレクトリ内の `.txt` ファイルを自動的に読み込み
-   **ベクトル化**: OpenAI の埋め込み API を使用してテキストをベクトル化
-   **類似検索**: Faiss を使用した高速な類似ドキュメント検索
-   **回答生成**: 検索結果をコンテキストとして使用して LLM で回答を生成

## 参考資料

-   [RAG をゼロから実装して仕組みを学ぶ【2025 年版】](https://zenn.dev/knowledgesense/articles/2619c6e5918d08)
-   [OpenAI API Documentation](https://platform.openai.com/docs)
-   [Streamlit Documentation](https://docs.streamlit.io/)
-   [Faiss Documentation](https://github.com/facebookresearch/faiss)

## ライセンス

このプロジェクトは学習目的で作成されています。
