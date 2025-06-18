from flask import Flask, request, jsonify, render_template, redirect, url_for, flash
from flask_login import login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from flask_cors import CORS
import os

from extensions import db, login_manager
from models import User, Image, Category

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB
app.config['SECRET_KEY'] = 'your_secret_key'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///site.db'
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 1024 * 1024 * 1024  # 1GB
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs('static/thumbs', exist_ok=True)
CORS(app)

db.init_app(app)
login_manager.init_app(app)
login_manager.login_view = 'login'

with app.app_context():
    db.create_all()

@app.route('/')
@login_required
def index():
    from models import User, Category
    page = request.args.get('page', 1, type=int)
    per_page = request.args.get('per_page', 50, type=int)
    category_id = request.args.get('category_id', type=int)
    user_id = request.args.get('user_id', type=int)
    start_date = request.args.get('start_date')
    end_date = request.args.get('end_date')

    query = Image.query
    if not current_user.is_admin:
        query = query.filter_by(user_id=current_user.id)

    if category_id:
        query = query.filter_by(category_id=category_id)
    if user_id:
        query = query.filter_by(user_id=user_id)
    if start_date:
        query = query.filter(Image.timestamp >= start_date)
    if end_date:
        query = query.filter(Image.timestamp <= end_date)

    query = query.order_by(Image.timestamp.desc())
    images = query.paginate(page=page, per_page=per_page)

    categories = Category.query.all()
    users = User.query.all()
    return render_template('index.html', images=images, categories=categories, users=users)
@login_required
def index():
    page = request.args.get('page', 1, type=int)
    images = Image.query.filter_by(user_id=current_user.id).order_by(Image.timestamp.desc()).paginate(page=page, per_page=8)
    return render_template('index.html', images=images)


@app.route('/upload', methods=['GET', 'POST'])
@login_required
def upload():
    print('[上传] 进入 upload() 路由')
    from werkzeug.utils import secure_filename
    import os
    from models import Image, Category
    from thumbnail_utils import generate_thumbnail

    if request.method == 'POST':
        files = request.files.getlist('image')
        print(f'[上传] 接收到文件数量: {len(files)}')
        category_id = request.form.get('category_id') or None
        if category_id == '':
            category_id = None

        has_valid_file = False

        for file in files:
            if not file or file.filename == '':
                print('[上传] 跳过空文件')
                continue

            has_valid_file = True

            filename = secure_filename(file.filename)
            upload_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(upload_path)

            # 生成缩略图
            thumb_path = os.path.join('static/thumbs', filename)
            generate_thumbnail(upload_path, os.path.join(os.getcwd(), thumb_path))

            new_image = Image(filename=filename, user_id=current_user.id, category_id=category_id)
            db.session.add(new_image)

        if has_valid_file:
            db.session.commit()
            flash("上传成功")
            return redirect(url_for('index'))
        else:
            flash("未选择任何有效图片")
            return redirect(url_for('upload'))

    categories = Category.query.all()
    return render_template('upload.html', categories=categories)
    categories = Category.query.all()
    return render_template('upload.html', categories=categories)
    return render_template('upload.html', categories=categories)

@app.route('/view/<int:image_id>')
@login_required
def view_image(image_id):
    image = Image.query.get_or_404(image_id)
    return render_template('view_image.html', image=image)

@app.route('/delete/<int:image_id>', methods=['POST'])
@login_required
def delete(image_id):
    image = Image.query.get_or_404(image_id)
    if image.user_id != current_user.id:
        return 'Unauthorized', 403
    try:
        os.remove(os.path.join(app.config['UPLOAD_FOLDER'], image.filename))
    except:
        pass
    db.session.delete(image)
    db.session.commit()
    return redirect(url_for('index'))

@app.route('/categories', methods=['GET', 'POST'])
@login_required
def categories():
    if request.method == 'POST':
        name = request.form['name']
        db.session.add(Category(name=name))
        db.session.commit()
        flash('分类添加成功')
        return redirect(url_for('categories'))
    categories = Category.query.all()
    return render_template('categories.html', categories=categories)

@app.route('/categories/delete/<int:id>', methods=['POST'])
@login_required
def delete_category(id):
    category = Category.query.get_or_404(id)
    db.session.delete(category)
    db.session.commit()
    return redirect(url_for('categories'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        user = User.query.filter_by(username=request.form['username']).first()
        if not user or not check_password_hash(user.password, request.form['password']):
            return "Invalid credentials", 401
        login_user(user)
        return redirect(url_for('index'))
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        existing_user = User.query.filter_by(username=username).first()
        if existing_user:
            return "用户已存在", 400
        is_admin = User.is_first_user()
        new_user = User(username=username, password=generate_password_hash(password), is_admin=is_admin)
        db.session.add(new_user)
        db.session.commit()
        return redirect(url_for('login'))
    return render_template('register.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

@app.route('/admin/users')
@login_required
def admin_users():
    if not current_user.is_admin:
        return "无权限访问", 403
    from sqlalchemy import func
    from models import User, Image
    users = User.query.all()
    for user in users:
        user.image_count = Image.query.filter_by(user_id=user.id).count()
    return render_template('admin_users.html', users=users)

@app.route('/admin/stats')
@login_required
def admin_stats():
    if not current_user.is_admin:
        return "无权限访问", 403
    from models import User, Image, Category
    from sqlalchemy import func
    user_count = User.query.count()
    image_count = Image.query.count()

    # 每日上传统计
    daily_data = db.session.query(func.date(Image.timestamp), func.count()).group_by(func.date(Image.timestamp)).all()
    daily_labels = [str(row[0]) for row in daily_data]
    daily_counts = [row[1] for row in daily_data]

    # 分类统计
    category_data = db.session.query(Category.name, func.count(Image.id))        .outerjoin(Image, Category.id == Image.category_id)        .group_by(Category.id).all()
    category_labels = [row[0] for row in category_data]
    category_counts = [row[1] for row in category_data]

    return render_template('admin_stats.html',
        user_count=user_count,
        image_count=image_count,
        daily_labels=daily_labels,
        daily_counts=daily_counts,
        category_labels=category_labels,
        category_counts=category_counts
    )

if __name__ == '__main__':
    app.run(debug=True)
@app.route('/batch_delete', methods=['POST'])
@login_required
def batch_delete():
    ids = request.form.getlist('image_ids')
    if not ids:
        flash("未选择图片")
        return redirect(url_for('index'))
    images = Image.query.filter(Image.id.in_(ids))
    for image in images:
        # 管理员可删所有，普通用户仅删自己图
        if current_user.is_admin or image.user_id == current_user.id:
            try:
                os.remove(os.path.join(app.config['UPLOAD_FOLDER'], image.filename))
                os.remove(os.path.join('static/thumbs', image.filename))
            except:
                pass
            db.session.delete(image)
    db.session.commit()
    flash("已删除选中图片")
    return redirect(url_for('index'))