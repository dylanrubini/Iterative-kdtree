!! author: Travis Sluka
!! category: support
!! Kd-tree creation and search methods

module kdtree_iterative
  !! A coords based 3D kd tree for fast retrieval of points.
  !!
  !! Initialization of the tree via kd_init() creates a 3d tree
  !! with points converted into x/y/z space. Once the tree is constructed
  !! points can then be retrieved in O(log n) time either by a point and
  !! max radius with kd_search_radius(), or by a point and
  !! the number of desired closest points to return kd_search_nnearest_iterative()
  !!
  !!
  !! @note Algorithm derived from Numerical Recipes, 2007
  !!
  !! @todo Allow for specification of MINDIV during initialization.
  !! Different trees might require different values here
  !!
  !! @todo fix bug where crashes on init if too few observations
  !!

  ! remove pure or elemental sub/func attribute because CUDA device kernel cannot have this   

  implicit none

  private


  ! public module methods
  !------------------------------------------------------------
  public :: kd_root, kd_init, kd_free
  public :: kd_search_radius, kd_search_nnearest_iterative


  ! private module parameters
  !------------------------------------------------------------
  integer, parameter :: dp=kind(0.0d0)
  integer, parameter :: sp=kind(0.0e0)

  integer, parameter :: kd_dim = 3
  !! 3 dimensions, x/y/z

  integer, parameter :: task_size = 50
  !! size of the tree division and search job stacks

  integer, parameter :: MINDIV = 10
  !! kd-tree will not bother dividing a box any further if it contains
  !! this many or fewer items.

  real(8), parameter :: pi = 4*atan(1.0_dp)
  !! pi, duh

  real(8), parameter :: re = 6371.3d3
  !! radius of earth


  ! custom types
  !------------------------------------------------------------
  type kd_root
     !! wrapper for all the information needed by the kd-tree after
     !! creation and during searching. Object is created by kd_init()
     !! to be used by kd_search_radius() and kd_search_nnearest_iterative().
     !! The User shouldn't need to access any of these members directly

     integer, pointer :: ptindx(:)
     !! array holding pointers to the coords indexes, this
     !! is the array that is organized by the creation of the kd tree

     type(boxnode), pointer :: boxes(:)
     !! collection of all boxes created for the kd tree

     real(8), pointer :: pts(:,:)
     !! array of all points stored in the kd tree, in x/y/z form

  end type kd_root



  type boxnode
     !! A node for defining each box used by the kd-tree.
     !! The end-user should normally be able to ignore what is in this object

     real(8) :: lo(kd_dim), hi(kd_dim)
     !! lower and higher bound of this box, in x/y/z space

     integer :: mom
     !! index of parent box

     integer :: dau1
     !! index of 1st daughter box, or 0 if no daughter

     integer :: dau2
     !! index of 1st daughter box, or 0 if no daughter

     integer :: ptlo
     !! index within "ptindx" of first point contained within thi box

     integer :: pthi
     !! index within "ptindx" of last point contained within thi box
  end type boxnode




contains


  
  !================================================================================
  !================================================================================


  
  subroutine kd_free(root)
    !! Frees up any resources associated with a kd tree created with kd_init

    type(KD_ROOT), intent(inout)  :: root

    deallocate(root%ptindx)
    deallocate(root%pts)
    deallocate(root%boxes)
  end subroutine kd_free


  
  !================================================================================
  !================================================================================

  

  subroutine kd_init(root, coords)
    !! Initialize a kd-tree structure given a list of coords
    !! coords variables are copied internally by the module and so
    !! can be deleted after calling this subroutine.

    type(KD_ROOT), intent(out)  :: root
    !! the root node of our kd-tree, used by kd_search_nnearest_iterative() and kd_search_radius()

    !! x, y, z coordinates (or cylindircal (x, r, rtheta) etc.)
    real(8), intent(in), target :: coords(:,:)

    ! variables used for kd-tree creation loop... too lazy to document what they all are, sorry.
    integer :: kk, np, ntmp, n, m, nboxes, nowtask, ptlo, pthi
    integer :: tmom, tdim, jbox
    real(8), pointer :: cp(:)
    integer,  pointer :: hp(:)
    real(8) :: lo(kd_dim), hi(kd_dim)
    integer  :: taskmom(50), taskdim(50)


    ! generate initial unsorted index array
    allocate(root%ptindx(size(coords,2)))
    do n=1, size(root%ptindx)
       root%ptindx(n) = n
    end do

    allocate(root%pts( kd_dim, size(coords, 2)))
    do n=1, size(coords, 2)
       root%pts(:,n)    = coords(1:kd_dim, n)
    end do

    ! calculate the number of kd boxes needed and create memory for them
    m = 1
    ntmp = size(root%ptindx)
    do while (ntmp > 0)
       ntmp = ishft(ntmp, -1)
       m = ishft(m,1)
    end do
    nboxes = 2*size(root%ptindx)-ishft(m, -1)
    if (m<nboxes) nboxes = m
    allocate(root%boxes(nboxes))

    ! initialize the root box, and put its subdivision on the task list
    lo = (/-1d20, -1d20, -1d20/)
    hi = (/ 1d20,  1d20,  1d20/)
    root%boxes(1) = boxnode(lo,hi,0,0,0,1,size(root%ptindx))

    !if we were give a small list, just quit now, there is nothing to divide
    if(size(root%ptindx) < MINDIV) return

    !otherwise start splitting up the tree
    jbox = 1
    taskmom(1) = 1
    taskdim(1) = 0
    nowtask = 1

    do while(nowtask > 0)
       ! get the box sitting on the top of the task list
       tmom = taskmom(nowtask)
       tdim = taskdim(nowtask)
       nowtask = nowtask - 1
       ptlo = root%boxes(tmom)%ptlo
       pthi = root%boxes(tmom)%pthi
       hp => root%ptindx(ptlo:pthi)

       ! rotate division among x/y/z coordinates
       cp => root%pts(tdim+1,:)

       ! determine dividing points
       np = pthi - ptlo + 1 ! total points
       kk = (np+1)/2        ! leftmost point of first subdivision

       ! do the array partitioning
       call kd_selecti(kk, hp, cp)

       ! create the daughters and push them onto the stack
       ! list if they need further subdivision
       hi = root%boxes(tmom)%hi
       lo = root%boxes(tmom)%lo
       lo(tdim+1) = cp(hp(kk))
       hi(tdim+1) = lo(tdim+1)
       root%boxes(jbox+1) = boxnode(root%boxes(tmom)%lo, hi, tmom, 0, 0, ptlo, ptlo+kk-1)
       root%boxes(jbox+2) = boxnode(lo, root%boxes(tmom)%hi, tmom, 0, 0, ptlo+kk, pthi)
       jbox = jbox+2
       root%boxes(tmom)%dau1 = jbox-1
       root%boxes(tmom)%dau2 = jbox

       ! subdivide the left further
       if (kk > MINDIV) then
          nowtask = nowtask + 1
          taskmom(nowtask) = jbox-1
          taskdim(nowtask) = mod(tdim+1, kd_dim)
       end if

       ! subdivide the right further
       if (np-kk > MINDIV+2) then
          nowtask = nowtask + 1
          taskmom(nowtask) = jbox
          taskdim(nowtask) = mod(tdim+1, kd_dim)
       end if
    end do
  end subroutine kd_init



  !================================================================================
  !================================================================================



  subroutine kd_search_radius(root, s_xyz, s_radius, r_points, r_distance, r_num)
    !! searches for all the points within a given radius.
    !! Maximum number of points to search for depends on the size of "r_points" and "r_distance".
    !! Once these arrays are full the subroutine will exit early.

    type(kd_root), intent(in) :: root
    !! root node containing all the information about the kd-tree

    real(8), intent(in) :: s_xyz(kd_dim)
    !! The longitude of the center of the search location (degrees)

    real(8), intent(in) :: s_radius
    !! the radius of the search (meters)

    integer, intent(out) :: r_points(:)
    !! the resulting list of points that are found,
    !! this is an array of indexes pointing to the original lat/lon arrays passed into kd_init().
    !! Array passed in should be the same size as "r_distance".

    real(8), intent(out) :: r_distance(:)
    !! the distance (meters) between each found point and the given search point.
    !! Array passed in should be the same size as "r_points"

    integer, intent(out) :: r_num
    !! the number of resulting points that were found.
    !! r_points[1:r_num] and r_distance[1:r_num] are populated after calling this subroutine

    !! if true, the exact great circle distances will be calculated (slower). Otherwise
    !! the euclidean distances are calculated (faster). The faster method
    !! is close enough for most purposes, especially if the search radius is small
    !! compared to the radius of the earth. Default is False.


    real(8) :: r
    integer  :: k, i, n, nb, nbold, ntask, jdim, d1, d2
    integer  :: task(task_size)
    type(boxnode), pointer :: boxes(:)

    boxes => root%boxes


    ! some basic checks
    if (size(r_points) /= size(r_distance)) then
       write (*,*) "ERROR: kd_search(), r_points and r_distance must be allocated with same size"
       stop 1
    end if

    ! find the smallest box that completely contains the bounds of the search point
    nb = 1
    jdim = 0
    do while(boxes(nb)%dau1 /= 0)
       nbold = nb
       d1 = boxes(nb)%dau1
       d2 = boxes(nb)%dau2
       if( s_xyz(jdim+1) + s_radius <=  boxes(d1)%hi(jdim+1)) then
          nb = d1
       else if ( s_xyz(jdim+1) - s_radius >= boxes(d2)%lo(jdim+1)) then
          nb = d2
       end if
       jdim = mod(jdim+1, kd_dim)
       if (nb == nbold) exit
    end do

    ! traverse the tree below this starting box, only as needed
    task(1) = nb
    ntask = 1
    r_num = 0

    do while(ntask /= 0)
       k = task(ntask)
       ntask = ntask - 1

       ! ignore boxes definitely outside the radius
       i = 0
       do n = 1, kd_dim
          if( (boxes(k)%lo(n) - (s_xyz(n)+s_radius)) * (boxes(k)%hi(n) - (s_xyz(n)-s_radius)) > 0) then
             i = 1
             exit
          end if
       end do
       if (i == 1) cycle

       if(boxes(k)%dau1 /= 0) then
          ! process child boxes
          task(ntask+1) = boxes(k)%dau1
          task(ntask+2) = boxes(k)%dau2
          ntask = ntask + 2
       else
          ! process points in this box
          do i = boxes(k)%ptlo, boxes(k)%pthi
             n = root%ptindx(i)

             ! calculate distance, either great-circle (slower) or
             ! Euclidean (faster)

             ! Euclidean distance, should be close enough to great-circle
             ! distance if the search radius is small enough, plus its much
             ! faster. Euclidean distance is smaller than greatcircle, and so more
             ! points will be included here
             r = dist_euc(s_xyz, root%pts(:,n))


             ! a new point was found
             if ( r <= s_radius) then
                !make sure there's room for the new found point
                if (r_num == size(r_points)) then
                   write (*,*) "ERROR: more points were found than there was room for in kd_search().",&
                        "Increase the size of the r_points and r_distance arrays"
                   return
                end if

                ! add the obs
                r_num = r_num + 1
                r_points(r_num) = n
                r_distance(r_num) = r
             end if

          end do
       end if
    end do
  end subroutine kd_search_radius



  !================================================================================
  !================================================================================



  subroutine kd_search_nnearest_iterative(root, s_xyz, s_num, r_points, r_distance, r_num)
    !! selects the "s_num" points in the kd tree that are nearest the search point.

    type(kd_root), intent(in) :: root
    !! root node containing all the information about the kd-tree

    real(8), intent(in) :: s_xyz(kd_dim)

    integer, intent(in) :: s_num
    !! the max number of points to find,

    integer, intent(out) :: r_points(:)
    !! the resulting list of points that are found,

    real(8), intent(out) :: r_distance(:)
    !! the distance (meters) between each found point and the given search point

    integer, intent(out) :: r_num
    !! the number of resulting points that were found

    !! if true, the exact great circle distances will be calculated (slower). Otherwise
    !! the euclidean distances are calculated (faster). The faster method
    !! is close enough for most purposes, especially if the search radius is small
    !! compared to the radius of the earth. Default is False.


    real(8) :: dn(s_num)
    !! heap, containing distances to points

    integer  :: nn(s_num)
    !! array if point indexes, paired with heap array entries

    integer  :: kp, i, n, ntask, k
    real(8)  :: d
    integer  :: task(task_size)

    ! set all entries in the heap to a really big number
    dn = 1d20

    ! find the smallest mother box with enough points to initialize the heap
    kp = kd_locate(root, s_xyz)
    do while(root%boxes(kp)%pthi-root%boxes(kp)%ptlo < s_num)
       kp = root%boxes(kp)%mom
    end do

    ! initialize the heap with the "s_num" closest points
    do i = root%boxes(kp)%ptlo, root%boxes(kp)%pthi
       n = root%ptindx(i)

       ! euclidean distance (faster)
       d = dist_euc( s_xyz, root%pts(:,n))

       ! if a closer point was found
       if (d < dn(1) ) then
          dn(1) = d
          nn(1) = n
          if (s_num > 1) call sift_down(dn, nn)
       end if
    end do

    ! traverse the tree opening possibly better boxes
    task(1) = 1
    ntask = 1
    do while (ntask /= 0)
       k = task(ntask)
       ntask = ntask - 1

       !dont reuse the box used to initialize the heap
       if (k == kp) cycle

       ! calculate min distance to point (via euclidean distance)
       ! this is still okay even if "exact" is true and we need to
       ! calculate great-circle distance for the points
       d = dist_box(root%boxes(k), s_xyz)

       if (d < dn(1)) then
          ! found a box with points potentially closer
          if (root%boxes(k)%dau1 /= 0) then
             ! put child boxes on task list
             task(ntask+1) = root%boxes(k)%dau1
             task(ntask+2) = root%boxes(k)%dau2
             ntask = ntask + 2
          else
             ! process points in this box
             do i=root%boxes(k)%ptlo, root%boxes(k)%pthi
                n = root%ptindx(i)

                ! euclidean distance
                d = dist_euc(root%pts(:,n), s_xyz)


                if (d < dn(1)) then
                   ! found a closer point, add it to the heap
                   dn(1) = d
                   nn(1) = n
                   if (s_num > 1) call sift_down(dn, nn)
                end if
             end do
          end if

       end if
    end do

    ! prepare output
    r_num = s_num
    ! TODO, handle situations where number of points found are less than number requested
    do n=1,s_num
       r_points(n) = nn(n)
       r_distance(n) = dn(n)
    end do

  end subroutine kd_search_nnearest_iterative



  !================================================================================
  !================================================================================



  function kd_locate(root, point)
    !! Given an arbitrary point, return the index of which kdtree box it is in.
    !! Private module function that shouldn't be called by the end user

    type(kd_root), intent(in) :: root
    !! kd tree to search in

    real(8), intent(in) :: point(3)
    !! point to search for, in x/y/z space

    integer :: kd_locate
    !! index of resulting kd tree box

    integer :: d1, jdim, nb
    nb = 1
    jdim = 0

    do while (root%boxes(nb)%dau1 /= 0)
       d1 = root%boxes(nb)%dau1
       if (point(jdim+1) <= root%boxes(d1)%hi(jdim+1)) then
          nb=d1
       else
          nb=root%boxes(nb)%dau2
       end if

       jdim = mod(jdim+1, kd_dim)
    end do

    kd_locate = nb
  end function kd_locate



  !================================================================================
  !================================================================================



  subroutine kd_selecti(k, indx, arr)
    !! Permutes indx[1...n] to make
    !!  arr[indx[1..k-1]] <= arr[indx[k]] <= arr[indx[k+1,n]]
    !!   the array "arr" is not modified
    !! Used in the array partitioning of kd_init()

    integer, intent(in) :: k
    integer, intent(inout) :: indx(:)
    real(8),intent(in) :: arr(:)

    integer :: i,ia,ir,j,l,mid
    real(8) :: a

    ! pulled from numerical recipes, 2007.
    ! minimal documentation, sorry.
    l=1
    ir=size(indx)
    do while(.true.)
       if (ir <= l+1) then
          if (ir == l+1 .and. arr(indx(ir)) < arr(indx(l))) &
               call swap(indx(l), indx(ir))
          exit
       else
          mid = (l+ir) / 2
          call swap(indx(mid), indx(l+1))
          if (arr(indx(l)) > arr(indx(ir)))   call swap(indx(l),indx(ir))
          if (arr(indx(l+1)) > arr(indx(ir))) call swap(indx(l+1),indx(ir))
          if (arr(indx(l)) > arr(indx(l+1)))  call swap(indx(l),indx(l+1))
          i=l+1
          j=ir
          ia=indx(l+1)
          a=arr(ia)
          do while(.true.)
             i = i +1
             do while(arr(indx(i)) < a)
                i = i + 1
             end do
             j = j-1
             do while(arr(indx(j)) > a)
                j = j - 1
             end do
             if (j < i) exit
             call swap(indx(i), indx(j))
          end do
          indx(l+1)=indx(j)
          indx(j)=ia
          if (j >= k) ir=j-1
          if (j<= k) l = i
       end if
    end do
  end subroutine kd_selecti



  !================================================================================
  !================================================================================



  pure subroutine swap(a1, a2)
    !! Convenience function to swap two array indices
    !! Used by kd_selecti()
    
    integer, intent(inout) :: a1, a2
    
    integer :: a
    
    a = a1
    a1 = a2
    a2 = a
  end subroutine swap



  !================================================================================
  !================================================================================



  pure function dist_euc(p1, p2)
    !! calculate the euclidean distance between two points

    real(8), intent(in) :: p1(kd_dim), p2(kd_dim)
    !! target points, in x/y/z space

    real(8) :: dist_euc
    !! euclidean distance (meters)

    integer :: n
    real(8) :: r1, r2

    dist_euc = SQRT( SUM( (p1 - p2)**2 ) )

  end function dist_euc



  !================================================================================
  !================================================================================



  pure function dist_box(box, p)
    !! if point is inside the box, return 0. Otherwise,
    !! caculate the distance between a point and closest spot on the box

    type(boxnode), intent(in) :: box

    real(8), intent(in) :: p(kd_dim)

    real(8) :: dist_box, dd
    integer :: n

    dd = 0
    do n=1,kd_dim
       if (p(n) < box%lo(n)) dd = dd + (p(n) - box%lo(n))*(p(n) - box%lo(n))
       if (p(n) > box%hi(n)) dd = dd + (p(n) - box%hi(n))*(p(n) - box%hi(n))
    end do

    dist_box = sqrt(dd)
  end function dist_box



  !================================================================================
  !================================================================================



  subroutine sift_down(heap, ndx)
    !! maintain a heap by sifting the first item down into its
    !! proper place. "ndx" is altered along with "heap"

    real(8), intent(inout) :: heap(:)
    !! the heap
    
    integer, intent(inout)  :: ndx(:)
    !! a second array whose elements are moved around to be kept
    !! matched with those of "heap"

    integer :: n, nn, j, jold, ia
    real(8) :: a

    nn = size(heap)
    n = nn
    a = heap(1)
    ia = ndx(1)
    jold = 1
    j = 2
    do while (j <= n)
       if ( j < n ) then
          if ( heap(j) < heap(j+1)) j = j+1
       end if
       if (a >= heap(j)) exit
       heap(jold) = heap(j)
       ndx(jold) = ndx(j)
       jold = j
       j = 2*j
    end do
    heap(jold) = a
    ndx(jold) = ia
  end subroutine sift_down



  !================================================================================

end module kdtree_iterative
