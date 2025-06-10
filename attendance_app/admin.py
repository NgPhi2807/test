from django.contrib import admin
from django.contrib.auth import get_user_model  # Lấy CustomUser model
from .models import CustomUser, Subject, SubjectStudent, SubjectDate, StudentAttendance

# Register CustomUser model
class CustomUserAdmin(admin.ModelAdmin):
    list_display = ('id','username', 'email', 'role', 'phone', 'avatar', 'date_created')
    search_fields = ('username', 'email', 'role')
    list_filter = ('role',)
    ordering = ('date_created',)

    def save_model(self, request, obj, form, change):
        if not change:  # Nếu là tạo mới người dùng
            obj.set_password(obj.password)  # Mã hóa mật khẩu
        else:
            # Nếu chỉnh sửa người dùng, chỉ mã hóa mật khẩu khi mật khẩu mới được thay đổi
            if obj.password != form.initial['password']:
                obj.set_password(obj.password)  # Mã hóa mật khẩu nếu thay đổi
        obj.save()  # Lưu người dùng

admin.site.register(CustomUser, CustomUserAdmin)


# Register Subject model
class SubjectAdmin(admin.ModelAdmin):
    list_display = ('name', 'teacher', 'time_start', 'time_end', 'date_start', 'date_end')
    search_fields = ('name', 'teacher__username')
    list_filter = ('teacher',)
    ordering = ('name',)

admin.site.register(Subject, SubjectAdmin)

# Register SubjectStudent model
class SubjectStudentAdmin(admin.ModelAdmin):
    list_display = ('subject', 'student')
    search_fields = ('subject__name', 'student__username')
    list_filter = ('subject', 'student')
    ordering = ('subject',)

admin.site.register(SubjectStudent, SubjectStudentAdmin)

# Register SubjectDate model
class SubjectDateAdmin(admin.ModelAdmin):
    list_display = ('id','subject', 'current_date', 'status')
    search_fields = ('subject__name',)
    list_filter = ('subject', 'status')
    ordering = ('subject', 'current_date')

admin.site.register(SubjectDate, SubjectDateAdmin)

# Register StudentAttendance model
class StudentAttendanceAdmin(admin.ModelAdmin):
    list_display = ('student', 'subject_date', 'status')
    search_fields = ('student__username', 'subject_date__subject__name')
    list_filter = ('status',)
    ordering = ('subject_date', 'student')

admin.site.register(StudentAttendance, StudentAttendanceAdmin)
