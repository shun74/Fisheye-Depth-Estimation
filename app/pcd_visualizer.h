#include <algorithm>
#include <atomic>
#include <mutex>
#include <omp.h>
#include <thread>

#include <opencv2/core.hpp>

#include <vtkActor.h>
#include <vtkAxes.h>
#include <vtkAxesActor.h>
#include <vtkCallbackCommand.h>
#include <vtkCamera.h>
#include <vtkCommand.h>
#include <vtkDataSetAttributes.h>
#include <vtkInteractorStyleTrackballCamera.h>
#include <vtkMaskPoints.h>
#include <vtkPointData.h>
#include <vtkPoints.h>
#include <vtkPolyData.h>
#include <vtkPolyDataMapper.h>
#include <vtkProperty.h>
#include <vtkRenderWindow.h>
#include <vtkRenderWindowInteractor.h>
#include <vtkRenderer.h>
#include <vtkSmartPointer.h>
#include <vtkUnsignedCharArray.h>
#include <vtkVertexGlyphFilter.h>

class PointCloudVisualizer
{
  public:
    PointCloudVisualizer(int width = 800, int height = 600, int max_points = 960 * 1280)
        : is_running_(false), cloud_updated_(false), max_points_(max_points)
    {
        points_ = vtkSmartPointer<vtkPoints>::New();
        points_->SetNumberOfPoints(max_points_);

        colors_ = vtkSmartPointer<vtkUnsignedCharArray>::New();
        colors_->SetNumberOfComponents(3);
        colors_->SetName("Colors");
        colors_->SetNumberOfTuples(max_points_);

        ghost_array_ = vtkSmartPointer<vtkUnsignedCharArray>::New();
        ghost_array_->SetNumberOfComponents(1);
        ghost_array_->SetName(vtkDataSetAttributes::GhostArrayName());
        ghost_array_->SetNumberOfTuples(max_points_);

        for (int i = 0; i < max_points_; ++i)
            ghost_array_->SetValue(i, vtkDataSetAttributes::DUPLICATEPOINT);

        poly_data_ = vtkSmartPointer<vtkPolyData>::New();
        poly_data_->SetPoints(points_);
        poly_data_->GetPointData()->SetScalars(colors_);
        poly_data_->GetPointData()->AddArray(ghost_array_);

        current_point_count_ = 0;

        vertex_filter_ = vtkSmartPointer<vtkVertexGlyphFilter>::New();
        vertex_filter_->SetInputData(poly_data_);

        vertex_filter_->SetInputArrayToProcess(0, 0, 0, vtkDataObject::FIELD_ASSOCIATION_POINTS,
                                               vtkDataSetAttributes::GhostArrayName());
        vertex_filter_->Update();

        mapper_ = vtkSmartPointer<vtkPolyDataMapper>::New();
        mapper_->SetInputConnection(vertex_filter_->GetOutputPort());

        actor_ = vtkSmartPointer<vtkActor>::New();
        actor_->SetMapper(mapper_);
        actor_->GetProperty()->SetPointSize(3);

        axes_actor_ = vtkSmartPointer<vtkAxesActor>::New();
        axes_actor_->SetTotalLength(0.1, 0.1, 0.1);
        axes_actor_->SetShaftType(0);
        axes_actor_->SetAxisLabels(0);

        renderer_ = vtkSmartPointer<vtkRenderer>::New();
        renderer_->AddActor(actor_);
        renderer_->AddActor(axes_actor_);
        renderer_->SetBackground(0.0, 0.0, 0.0);

        render_window_ = vtkSmartPointer<vtkRenderWindow>::New();
        render_window_->AddRenderer(renderer_);
        const std::string window_name = "Point Cloud";
        render_window_->SetWindowName(window_name.c_str());
        render_window_->SetSize(width, height);

        interactor_ = vtkSmartPointer<vtkRenderWindowInteractor>::New();
        interactor_->SetRenderWindow(render_window_);

        renderer_->GetActiveCamera()->SetPosition(0, 0, 2);
        renderer_->GetActiveCamera()->SetFocalPoint(0, 0, 0);
        renderer_->GetActiveCamera()->SetViewUp(0, 1, 0);

        vtkSmartPointer<vtkInteractorStyleTrackballCamera> style =
            vtkSmartPointer<vtkInteractorStyleTrackballCamera>::New();
        interactor_->SetInteractorStyle(style);

        timer_callback_ = vtkSmartPointer<vtkCallbackCommand>::New();
        timer_callback_->SetCallback(PointCloudVisualizer::TimerCallbackFunction);
        timer_callback_->SetClientData(this);
    }

    void start()
    {
        if (is_running_)
            return;

        is_running_ = true;
        viewer_thread_ = std::thread(&PointCloudVisualizer::visualizerThreadFunction, this);
    }

    void stop()
    {
        if (!is_running_)
            return;

        is_running_ = false;
        if (interactor_)
        {
            interactor_->TerminateApp();
        }

        if (viewer_thread_.joinable())
        {
            viewer_thread_.join();
        }
    }

    void updatePointCloud(cv::InputArray _cloud, cv::InputArray _colors, std::vector<bool> &valid)
    {
        cv::Mat cloud = _cloud.getMat();
        const cv::Point3f *cloud_ptr = cloud.ptr<cv::Point3f>();
        cv::Mat colors = _colors.getMat();
        const cv::Vec3b *colors_ptr = colors.ptr<cv::Vec3b>();

        std::lock_guard<std::mutex> lock(cloud_mutex_);

        int num_points = std::min(cloud.size().width, max_points_);

        float *point_data = static_cast<float *>(points_->GetVoidPointer(0));
        unsigned char *color_data = static_cast<unsigned char *>(colors_->GetVoidPointer(0));
        unsigned char *ghost_data = static_cast<unsigned char *>(ghost_array_->GetVoidPointer(0));

#pragma omp parallel for
        for (int i = 0; i < num_points; ++i)
        {
            int point_offset = i * 3;
            point_data[point_offset] = cloud_ptr[i].x;
            point_data[point_offset + 1] = cloud_ptr[i].y;
            point_data[point_offset + 2] = cloud_ptr[i].z;

            color_data[point_offset] = colors_ptr[i][0];
            color_data[point_offset + 1] = colors_ptr[i][1];
            color_data[point_offset + 2] = colors_ptr[i][2];

            if (valid[i])
                ghost_data[i] = 0;
            else
                ghost_data[i] = vtkDataSetAttributes::DUPLICATEPOINT;
        }

        if (num_points < max_points_)
        {
#pragma omp parallel for
            for (int i = num_points; i < max_points_; ++i)
            {
                ghost_data[i] = vtkDataSetAttributes::DUPLICATEPOINT;
            }
        }

        current_point_count_ = num_points;

        points_->Modified();
        colors_->Modified();
        ghost_array_->Modified();
        poly_data_->Modified();
        cloud_updated_ = true;
    }

    void setDownsamplingFactor(int factor)
    {
        if (factor <= 0)
            factor = 1;
        downsampling_factor_ = factor;

        applyDownsampling();
    }

    void setPointSize(int size)
    {
        if (actor_)
        {
            actor_->GetProperty()->SetPointSize(size);
        }
    }

    void spinOnce()
    {
        if (cloud_updated_)
        {
            std::lock_guard<std::mutex> lock(cloud_mutex_);
            vertex_filter_->Update();
            render_window_->Render();
            cloud_updated_ = false;
        }
    }

  private:
    vtkSmartPointer<vtkPoints> points_;
    vtkSmartPointer<vtkUnsignedCharArray> colors_;
    vtkSmartPointer<vtkUnsignedCharArray> ghost_array_;
    vtkSmartPointer<vtkPolyData> poly_data_;
    vtkSmartPointer<vtkVertexGlyphFilter> vertex_filter_;
    vtkSmartPointer<vtkPolyDataMapper> mapper_;
    vtkSmartPointer<vtkActor> actor_;
    vtkSmartPointer<vtkAxesActor> axes_actor_;
    vtkSmartPointer<vtkRenderer> renderer_;
    vtkSmartPointer<vtkRenderWindow> render_window_;
    vtkSmartPointer<vtkRenderWindowInteractor> interactor_;
    vtkSmartPointer<vtkCallbackCommand> timer_callback_;

    std::atomic<bool> is_running_;
    std::atomic<bool> cloud_updated_;
    int max_points_;
    int current_point_count_;
    int downsampling_factor_ = 1;
    std::thread viewer_thread_;
    std::mutex cloud_mutex_;

    void applyDownsampling()
    {
        std::lock_guard<std::mutex> lock(cloud_mutex_);

        if (downsampling_factor_ <= 1)
        {
            vertex_filter_->SetInputData(poly_data_);
        }
        else
        {
            vtkSmartPointer<vtkMaskPoints> mask_points = vtkSmartPointer<vtkMaskPoints>::New();
            mask_points->SetInputData(poly_data_);
            mask_points->SetOnRatio(downsampling_factor_);
            mask_points->SetRandomMode(false);
            vertex_filter_->SetInputConnection(mask_points->GetOutputPort());
        }

        vertex_filter_->Update();
        cloud_updated_ = true;
    }

    void visualizerThreadFunction()
    {
        interactor_->Initialize();
        interactor_->CreateRepeatingTimer(30);
        interactor_->AddObserver(vtkCommand::TimerEvent, timer_callback_);

        interactor_->Start();
    }

    static void TimerCallbackFunction(vtkObject *caller, long unsigned int event_id, void *client_data, void *call_data)
    {
        PointCloudVisualizer *self = static_cast<PointCloudVisualizer *>(client_data);

        if (self->cloud_updated_)
        {
            std::lock_guard<std::mutex> lock(self->cloud_mutex_);
            self->vertex_filter_->Update();
            self->render_window_->Render();
            self->cloud_updated_ = false;
        }
    }
};
